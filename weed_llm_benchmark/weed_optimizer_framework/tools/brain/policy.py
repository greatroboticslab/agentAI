"""The one gate a cluster action or a scheduler step passes through (v3.27.0).

Why this exists
----------------
Nothing before this module stood between "a model decided to do something" and
"the shared cluster account did it". `dashboard_server.py`'s `_CLUSTER_ACTIONS`
is whitelisted by name and RBAC-gated by who is logged in, but a logged-in
actor may fire any action in the table, at any parameter value the type
coercion in `api_cluster_action` happens to accept, with no notion that a
dataset delete and a cache refresh are different kinds of request. As tier-0
and tier-1 models start proposing these same actions (`TIERED_SUPERVISION_PLAN`
S4+), that gap becomes the whole risk surface: an escalation authority has to
sit where the request is decided, not where it is logged afterward.

What this module is
--------------------
A pure decision function, `authorize()`, over a data table
(`policy_actions.json`, loaded and validated once, cached on mtime): given who
is asking, what they are asking for, and with what parameters, it returns
whether the request may proceed at all, and if so, whether a human has to sign
off first. It does not submit anything, does not talk to Mongo or SLURM, and
never touches the string it is asked to authorize as anything but data —
`ACTIONS[...]["template"]` is descriptive metadata for an integrator to render
elsewhere; nothing in this file calls `eval`, `exec`, `subprocess` or any
templating engine on a table value.

The risk model — what R0..R4 mean, which actor-tier prefixes exist, and the
ceiling table (which tier may apply / propose / is refused at which risk) — is
CODE below, not data in the JSON file. `db.py` makes the same choice for
`_STEP_ALLOWED_SCRIPTS` and says why: a config a project owner can patch over
HTTP must never be able to grant itself more authority than the code allows.
If the ceiling table lived in JSON, editing that file would be a privilege
escalation path. Only the action catalogue — which actions exist, their
bounds, their cost, who may even ask — is data.

Fail-closed, structurally
--------------------------
There is no line in `authorize()` that returns `allowed: True` without having
matched both a known, well-formed action row AND a known actor-tier prefix
against an explicit ceiling-table entry for that tier at that risk. Every
other path — a malformed actor string, an unknown or malformed action id, a
param with no declared bound, a param outside its bound, an internal
exception of any kind — returns `allowed: False` before reaching that match.
The public entry point wraps its own body in one broad exception handler for
exactly this reason: a bundle of untrusted parameters raising out of a bound
check must fail the request, not crash the caller (`round_scheduler`'s tick
loop or the dashboard's request handler) into some other failure mode.

Risk tiers (WP5 spec, verbatim)
--------------------------------
    R0  read-only.                                            all actors, direct.
    R1  reversible config change, in bounds.                  tier-0 proposes, tier-1 applies.
    R2  compute or corpus state (train submit, cancel/        tier-1 applies (after an
        resubmit own step, quarantine with reason).           artifact-cited verdict).
    R3  external side effect, SU-heavy, or design-changing.    approval queue.
    R4  destructive or irreversible.                           human only.

Actor-tier ceilings (verbatim): tier-0 <= R1 propose, tier-1 <= R2 applies +
propose R3, tier-2 proposes only, human all. `round-scheduler` is not one of
the spec's four model/human tiers; it is the unattended loop's own identity
when `brain.policy == "scripted"` (no model decides anything, db.py
DEFAULT_DOMAIN_CONFIG). It is given the same direct authority as tier-1 up to
R2, because that is the authority the loop already exercises today submitting
collect/filter/train every round with no review — and nothing past R2, because
it is not a reviewer and a queued R3 proposal from a loop with no one reading
its queue would simply stall forever, which is worse than refusing it outright.

Actor strings (verbatim from the spec): `tier0:<model>`, `tier1:<model>`,
`tier2:<backend>`, `round-scheduler`, `human:<email>`. Only the prefix before
the first `:` decides the tier; the identity after it is carried through for
logging and is never itself inspected for a tier keyword — see `_ACTOR_RE`.

CLI:
    python -m weed_optimizer_framework.tools.brain.policy table
    python -m weed_optimizer_framework.tools.brain.policy explain <action_id>
    python -m weed_optimizer_framework.tools.brain.policy authorize <actor> <action> \
        [--params '<json>'] [--budget '<json>'] [--resources '<json>']
"""
import json
import math
import os
import pathlib
import re

TOOL_VERSION = "wp5-policy/1"

RISKS = ("R0", "R1", "R2", "R3", "R4")

# The five actor-tier prefixes this module recognises. Anything else is an
# unknown actor, refused before any table lookup happens.
TIERS = ("tier0", "tier1", "tier2", "round-scheduler", "human")

# One well-formed actor string, end to end. Deliberately strict: the whole
# string must fullmatch, not merely contain a known prefix somewhere in it.
# This is what makes an actor like "tier1:x human:harry" fail closed as an
# unrecognised actor rather than being read as "human" by a parser that scans
# for the word — the embedded space is outside the identity character class,
# so the regex simply does not match and the request is refused before any
# tier logic runs.
_ACTOR_RE = re.compile(
    r"^(?P<tier>tier0|tier1|tier2|human|round-scheduler)"
    r"(?::(?P<ident>[A-Za-z0-9._+@/-]{1,128}))?$"
)

# A table action id, and the request-supplied action string, must both look
# like this before a dict lookup is even attempted. Real ids in the table are
# lower/upper-case words with underscores (e.g. the real, mixed-case
# "audit_registry_garbage_APPLY"); nothing resembling a shell metacharacter,
# a path separator or whitespace is a valid action id, so a string carrying
# one is refused on shape alone, with a reason that names the defect, rather
# than falling through to "not in ACTIONS" which would be true but less
# useful to a human reading the refusal.
_ACTION_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")

# A cap on any string-typed parameter's raw length before its bound's pattern
# is even tried. The patterns in the table are small, fixed and written by
# this module's own author, not attacker-controlled, so this is not a defence
# against a hostile pattern — it is a cheap backstop against a pathological
# value on an ordinary pattern, and costs nothing on the common case.
_MAX_STR_PARAM_LEN = 4096

_TABLE_FILE = pathlib.Path(__file__).resolve().parent / "policy_actions.json"
_TABLE_CACHE = {"key": None, "actions": {}, "rates": {}, "errors": [], "path": ""}


class _PolicyDefect(Exception):
    """The table, an actor string or a param set is malformed. Always caught;
    never escapes `authorize()`. Distinguished from a bound plain 'refused'
    only so `_run_authorize` has one thing to catch around the whole body."""


# --- table loading -----------------------------------------------------------
def _table_path():
    return pathlib.Path(os.environ.get("BRAIN_POLICY_ACTIONS") or _TABLE_FILE)


def _validate_bound(name, bound):
    """(ok, reason) for one param_bounds entry. Never raises."""
    if not isinstance(bound, dict):
        return False, "bound for %r is not an object" % name
    kind = bound.get("type")
    if kind in ("int", "float"):
        lo, hi = bound.get("min"), bound.get("max")
        if not isinstance(lo, (int, float)) or isinstance(lo, bool):
            return False, "bound for %r has no numeric 'min'" % name
        if not isinstance(hi, (int, float)) or isinstance(hi, bool):
            return False, "bound for %r has no numeric 'max'" % name
        if lo > hi:
            return False, "bound for %r has min > max" % name
        return True, ""
    if kind == "enum":
        vt = bound.get("value_type")
        if vt not in ("int", "str"):
            return False, "bound for %r has no valid 'value_type'" % name
        values = bound.get("values")
        if not isinstance(values, list) or not values:
            return False, "bound for %r has no non-empty 'values' list" % name
        want = int if vt == "int" else str
        for v in values:
            if want is int:
                if not isinstance(v, int) or isinstance(v, bool):
                    return False, "bound for %r declares value_type int but carries %r" % (name, v)
            else:
                if not isinstance(v, str):
                    return False, "bound for %r declares value_type str but carries %r" % (name, v)
        return True, ""
    if kind == "str":
        pattern = bound.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            return False, "bound for %r has no 'pattern'" % name
        try:
            re.compile(pattern)
        except re.error as exc:
            return False, "bound for %r has an unparsable pattern: %s" % (name, exc)
        return True, ""
    return False, "bound for %r has unknown type %r" % (name, kind)


def _validate_est_su(action_id, formula, rates):
    """(ok, reason) for one est_su formula. Structural only — never computes."""
    if not isinstance(formula, dict):
        return False, "est_su is not an object"
    if not isinstance(formula.get("why"), str) or not formula["why"].strip():
        return False, "est_su carries no 'why'"
    if "fixed_su" in formula:
        v = formula["fixed_su"]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
            return False, "fixed_su must be a non-negative number"
        return True, ""
    if formula.get("unknown") is True:
        return True, ""
    gpu_type = formula.get("gpu_type")
    if not isinstance(gpu_type, str) or gpu_type not in rates:
        return False, ("est_su gpu_type %r has no rate in su_rates.json"
                       % gpu_type)
    has_count = ("gpu_count" in formula) != ("gpu_count_param" in formula)
    has_hours = ("hours" in formula) != ("hours_param" in formula)
    if not has_count:
        return False, "est_su must declare exactly one of gpu_count / gpu_count_param"
    if not has_hours:
        return False, "est_su must declare exactly one of hours / hours_param"
    if "gpu_count" in formula:
        gc = formula["gpu_count"]
        if not isinstance(gc, (int, float)) or isinstance(gc, bool) or gc <= 0:
            return False, "est_su gpu_count must be a positive number"
    if "hours" in formula:
        h = formula["hours"]
        if not isinstance(h, (int, float)) or isinstance(h, bool) or h < 0:
            return False, "est_su hours must be a non-negative number"
    if "hours_default" in formula:
        hd = formula["hours_default"]
        if not isinstance(hd, (int, float)) or isinstance(hd, bool) or hd < 0:
            return False, "est_su hours_default must be a non-negative number"
    return True, ""


_REQUIRED_ROW_KEYS = ("template", "param_bounds", "risk", "reversible",
                      "est_su", "dry_run_variant", "allowed_tiers", "description")


def _validate_row(action_id, row, rates):
    """(ok, reason) for one ACTIONS row. A failure here takes down only this
    action id — `_table()` isolates one bad row from the rest of the table,
    the same way `signals._run_check` isolates one bad check."""
    if not isinstance(row, dict):
        return False, "row is not an object"
    missing = [k for k in _REQUIRED_ROW_KEYS if k not in row]
    if missing:
        return False, "row is missing key(s): %s" % ", ".join(missing)
    if not isinstance(row["template"], str) or not row["template"].strip():
        return False, "template must be a non-empty string"
    if not isinstance(row["description"], str) or not row["description"].strip():
        return False, "description must be a non-empty string"
    if row["risk"] not in RISKS:
        return False, "risk %r is not one of %s" % (row["risk"], RISKS)
    if not isinstance(row["reversible"], bool):
        return False, "reversible must be a boolean"
    bounds = row["param_bounds"]
    if not isinstance(bounds, dict):
        return False, "param_bounds must be an object"
    for name, bound in bounds.items():
        ok, why = _validate_bound(name, bound)
        if not ok:
            return False, why
    ok, why = _validate_est_su(action_id, row["est_su"], rates)
    if not ok:
        return False, why
    dv = row["dry_run_variant"]
    if dv is not None and not isinstance(dv, str):
        return False, "dry_run_variant must be a string or null"
    tiers = row["allowed_tiers"]
    if not isinstance(tiers, list) or not tiers:
        return False, "allowed_tiers must be a non-empty list"
    for t in tiers:
        if t not in TIERS:
            return False, "allowed_tiers carries unknown tier %r" % t
    return True, ""


class _RateTable(object):
    """SU-per-GPU-hour, resolved through `su_ledger`, which owns the rates.

    This module used to carry its own copy of the H100 and V100 rates in
    `policy_actions.json`. Two files stating the same pre-registered number is
    the drift the threshold tables were externalised to prevent: an operator who
    corrects one and not the other leaves the estimate that gates approvals
    disagreeing with the ledger that reports the spend, and neither says so.

    A family the shared table cannot resolve is absent here rather than charged
    at a fallback. The ledger's fallback exists so recorded spend is never
    understated; an estimate used to decide whether an action may run must
    instead refuse, because approving against a guessed price is the failure
    this gate exists to prevent.
    """

    def __init__(self):
        self._cache = {}

    def _resolve(self, name):
        key = str(name)
        if key in self._cache:
            return self._cache[key]
        rate = None
        try:
            from . import su_ledger
            got = su_ledger.su_for(key, 1, 3600.0)
            if isinstance(got, dict) and not got.get("unknown_rate") \
                    and isinstance(got.get("value"), (int, float)):
                rate = float(got["value"])
        except Exception:
            rate = None
        self._cache[key] = rate
        return rate

    def get(self, name, default=None):
        rate = self._resolve(name)
        return default if rate is None else rate

    def __contains__(self, name):
        return self._resolve(name) is not None

    def keys(self):
        """The GPU families the shared table declares, fallback excluded.

        The fallback is deliberately not a key: it is the ledger's guard against
        understating recorded spend, and an estimate that gates an approval must
        never quietly price an unknown family off it.
        """
        out = []
        try:
            from . import su_ledger
            for key in (su_ledger.rates().get("values") or {}):
                parts = str(key).split(".")
                if len(parts) == 3 and parts[0] == "rates" \
                        and parts[1] != "unknown_gpu_fallback":
                    out.append(parts[1])
        except Exception:
            return []
        return sorted(set(out))

    def __iter__(self):
        return iter(self.keys())

    def __getitem__(self, name):
        rate = self._resolve(name)
        if rate is None:
            raise KeyError(name)
        return rate

    def items(self):
        return [(k, self._resolve(k)) for k in self.keys()]


def _load_rates(raw):
    """The shared rate table. `raw` is accepted and ignored: rates are not data
    of this file any more, and a `gpu_su_per_hour` block left behind in the
    action catalogue would be silently ineffective, so `_validate_table` refuses
    one outright rather than letting it look authoritative."""
    return _RateTable()


def _table():
    """The loaded, per-row-validated table. Cached on (path, mtime, size), the
    same convention `signals._file_block` uses, so an always-on dashboard
    process picks up an edited table on the next call with no restart.

    Never raises: a missing or unparsable file yields an empty action set plus
    a recorded error, which makes every subsequent `authorize()` call refuse
    for "unknown action" rather than crash — the file failing to load is not
    a reason to fail open."""
    path = _table_path()
    try:
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
    except OSError as exc:
        _TABLE_CACHE.update({"key": None, "actions": {}, "rates": {},
                             "path": str(path),
                             "errors": ["policy table unreadable at %s (%s)"
                                        % (path.name, type(exc).__name__)]})
        return _TABLE_CACHE
    if _TABLE_CACHE["key"] == key:
        return _TABLE_CACHE
    actions, errors = {}, []
    try:
        with open(str(path), "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception as exc:
        _TABLE_CACHE.update({"key": key, "actions": {}, "rates": {},
                             "path": str(path),
                             "errors": ["policy table unreadable: %s" % type(exc).__name__]})
        return _TABLE_CACHE
    rates = _load_rates(raw if isinstance(raw, dict) else {})
    rows = (raw or {}).get("actions")
    if not isinstance(rows, dict):
        errors.append("policy table carries no 'actions' object")
        rows = {}
    for action_id, row in rows.items():
        if not _ACTION_ID_RE.match(str(action_id)):
            errors.append("action id %r does not match the allowed shape; dropped" % action_id)
            continue
        ok, why = _validate_row(action_id, row, rates)
        if not ok:
            errors.append("action %r is malformed and was dropped: %s" % (action_id, why))
            continue
        actions[action_id] = row
    # cross-validate dry_run_variant references once every row is known good
    for action_id, row in list(actions.items()):
        dv = row.get("dry_run_variant")
        if dv is not None and dv not in actions:
            errors.append("action %r declares dry_run_variant %r, which is not a "
                          "valid action; dropped" % (action_id, dv))
            del actions[action_id]
    _TABLE_CACHE.update({"key": key, "actions": actions, "rates": rates,
                         "path": str(path), "errors": errors})
    return _TABLE_CACHE


# --- actor-tier ceiling (CODE, not data — see module docstring) --------------
# decision per (tier, risk): "direct" (allowed, no approval needed), "propose"
# (allowed, needs_approval), or absent, which _decide() treats as "refuse".
# Absent is the default for every cell this table does not list, which is what
# makes an R4 request from anyone but human fail closed with no special-casing.
_CEILING = {
    "tier0":           {"R0": "direct", "R1": "propose"},
    "tier1":           {"R0": "direct", "R1": "direct", "R2": "direct", "R3": "propose"},
    "tier2":           {"R0": "direct", "R1": "propose", "R2": "propose", "R3": "propose"},
    "round-scheduler": {"R0": "direct", "R1": "direct", "R2": "direct"},
    "human":           {"R0": "direct", "R1": "direct", "R2": "direct", "R3": "direct", "R4": "direct"},
}


def _decide(tier, risk):
    return _CEILING.get(tier, {}).get(risk, "refuse")


# --- param bound checking ----------------------------------------------------
def _check_value(name, value, bound):
    """(ok, reason) for one supplied param against its declared bound.

    No coercion, ever: a string that looks like a number is refused, not
    parsed, because the whole point of a declared type is that the caller
    already agrees what type it is sending. `bool` is excluded from the
    numeric types explicitly — `isinstance(True, int)` is true in Python, and
    accepting it would let a boolean silently stand in for 0/1.
    """
    kind = bound["type"]
    if kind == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            return False, "%r must be an int, got %s" % (name, type(value).__name__)
        if not (bound["min"] <= value <= bound["max"]):
            return False, "%r=%s is outside [%s, %s]" % (name, value, bound["min"], bound["max"])
        return True, ""
    if kind == "float":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False, "%r must be a number, got %s" % (name, type(value).__name__)
        if math.isnan(value) or math.isinf(value):
            return False, "%r is NaN or infinite" % name
        if not (bound["min"] <= value <= bound["max"]):
            return False, "%r=%s is outside [%s, %s]" % (name, value, bound["min"], bound["max"])
        return True, ""
    if kind == "enum":
        want = int if bound["value_type"] == "int" else str
        if want is int:
            if not isinstance(value, int) or isinstance(value, bool):
                return False, "%r must be an int, got %s" % (name, type(value).__name__)
        else:
            if not isinstance(value, str):
                return False, "%r must be a string, got %s" % (name, type(value).__name__)
        if value not in bound["values"]:
            return False, "%r=%r is not one of %s" % (name, value, bound["values"])
        return True, ""
    if kind == "str":
        if not isinstance(value, str):
            return False, "%r must be a string, got %s" % (name, type(value).__name__)
        if len(value) > _MAX_STR_PARAM_LEN:
            return False, "%r is longer than %d characters" % (name, _MAX_STR_PARAM_LEN)
        if not re.fullmatch(bound["pattern"], value):
            return False, "%r does not match the required pattern" % name
        return True, ""
    return False, "%r has an unrecognised bound type %r" % (name, kind)   # unreachable: _table() drops this row


def _check_params(params, bounds):
    """(ok, [reasons]) for a whole params dict against one action's bounds.

    Every key in `params` must have a declared bound — an undeclared key is
    refused as a missing bound declaration, which is the same failure mode as
    an extra, unrecognised key: the table is the only source of truth for
    which parameters an action accepts at all, so anything it does not name is
    unauthorized by construction, not merely unchecked.
    """
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return False, ["params must be an object, got %s" % type(params).__name__]
    reasons = []
    for key in params:
        if not isinstance(key, str):
            reasons.append("param key %r is not a string" % (key,))
            continue
        bound = bounds.get(key)
        if bound is None:
            reasons.append("param %r has no declared bound for this action; refused" % key)
            continue
        ok, why = _check_value(key, params[key], bound)
        if not ok:
            reasons.append(why)
    return (not reasons), reasons


# --- public read helpers ------------------------------------------------------
def risk_of(action):
    """The risk tier of `action`, or None if it is unknown or malformed."""
    row = _table()["actions"].get(str(action))
    return row["risk"] if row else None


def describe(action):
    """The full table row for `action`, plus its id, or a `known: False`
    stub naming why it cannot be described. Never raises."""
    action = str(action)
    row = _table()["actions"].get(action)
    if row is None:
        return {"action": action, "known": False,
                "reason": "not a recognised action (unknown id, or its table "
                          "row failed validation — see errors())"}
    out = dict(row)
    out["action"] = action
    out["known"] = True
    return out


def actions_for_tier(tier):
    """Action ids `tier` may request at all (direct or propose), sorted.

    Informational — a menu for a UI or a model's own tool list, not a
    substitute for calling `authorize()` on the actual request, which also
    checks the request's parameters and the live budget/resource state.
    """
    tier = str(tier)
    out = []
    for action_id, row in _table()["actions"].items():
        if tier not in row["allowed_tiers"]:
            continue
        if _decide(tier, row["risk"]) != "refuse":
            out.append(action_id)
    return sorted(out)


def estimate_su(action, params=None):
    """{"action", "su": float|None, "confident": bool, "reason": str}.

    `su` is None when the table honestly does not know the cost (an unknown
    action, or a formula the table marks `unknown: true`) — never a guessed
    number standing in for a real one. Reading a missing `hours_param` key as
    0 would be exactly that guess, so it is reported unknown instead.
    """
    action = str(action)
    row = _table()["actions"].get(action)
    if row is None:
        return {"action": action, "su": None, "confident": False,
                "reason": "not a recognised action"}
    formula = row["est_su"]
    params = params if isinstance(params, dict) else {}
    if "fixed_su" in formula:
        return {"action": action, "su": float(formula["fixed_su"]),
                "confident": True, "reason": formula["why"]}
    if formula.get("unknown") is True:
        return {"action": action, "su": None, "confident": False,
                "reason": formula["why"]}
    rates = _table()["rates"]
    rate = rates.get(formula["gpu_type"])
    if rate is None:
        return {"action": action, "su": None, "confident": False,
                "reason": "gpu_type %r has no declared SU-per-hour rate"
                          % formula["gpu_type"]}
    if "gpu_count" in formula:
        gpu_count = formula["gpu_count"]
    else:
        gpu_count = params.get(formula["gpu_count_param"])
        if not isinstance(gpu_count, (int, float)) or isinstance(gpu_count, bool):
            return {"action": action, "su": None, "confident": False,
                    "reason": "params carries no numeric %r; gpu_count is "
                              "not guessed" % formula["gpu_count_param"]}
    if "hours" in formula:
        hours = formula["hours"]
    else:
        hours = params.get(formula["hours_param"], formula.get("hours_default"))
        if not isinstance(hours, (int, float)) or isinstance(hours, bool):
            return {"action": action, "su": None, "confident": False,
                    "reason": "params carries no numeric %r and the action "
                              "declares no default; hours is not guessed"
                              % formula["hours_param"]}
    su = round(float(gpu_count) * float(hours) * rate, 3)
    return {"action": action, "su": su, "confident": True,
            "reason": "%s x %s GPU-h @ %s SU/GPU-h (%s)"
                      % (gpu_count, hours, rate, formula["why"])}


def errors():
    """Table-load and per-row validation errors from the last `_table()` call.
    Empty on a clean table. Exposed so a health check can alarm on a policy
    table that is silently dropping rows rather than discovering it only when
    an action nobody removed on purpose stops being authorizable."""
    return list(_table()["errors"])


# --- authorize -----------------------------------------------------------------
def _result(allowed, needs_approval, risk, reasons):
    """The one place a result dict is built, so the invariant that
    needs_approval only means something when allowed is True cannot be broken
    by a call site forgetting to enforce it."""
    return {"allowed": bool(allowed),
            "needs_approval": bool(needs_approval) if allowed else False,
            "risk": risk,
            "reasons": [str(r) for r in reasons]}


def _parse_actor(actor):
    """(tier, identity) or raises _PolicyDefect naming the defect."""
    if not isinstance(actor, str) or not actor:
        raise _PolicyDefect("actor must be a non-empty string, got %s"
                            % type(actor).__name__)
    m = _ACTOR_RE.match(actor)
    if not m:
        raise _PolicyDefect(
            "actor %r is not a recognized tier prefix (expected tier0:<model>, "
            "tier1:<model>, tier2:<backend>, round-scheduler, or "
            "human:<email>)" % actor)
    return m.group("tier"), (m.group("ident") or "")


def _parse_action_id(action):
    if not isinstance(action, str) or not action:
        raise _PolicyDefect("action must be a non-empty string, got %s"
                            % type(action).__name__)
    if not _ACTION_ID_RE.match(action):
        raise _PolicyDefect(
            "action id %r contains characters outside A-Z a-z 0-9 _, or is "
            "too long; refused before any table lookup" % action)
    return action


def _check_resources(resources, risk):
    """A reason to refuse regardless of tier, or "" when resources say nothing.

    `mongo_down` is the one fact this checks, and it is grounded directly in
    the executor handbook's write path: "a failed Mongo write raises the
    mongo_down signal and blocks submissions" (WP5 shared conventions). A
    request this module allowed but the ledger cannot record would be exactly
    the 2026-08-29 blind spot again — the loop keeps acting and nothing
    remembers it acted — so R1 and above are refused outright while Mongo is
    down. R0 is exempt: a read records nothing and blocks on nothing.
    """
    if risk == "R0" or not isinstance(resources, dict):
        return ""
    if resources.get("mongo_down"):
        return ("mongo_down: the round ledger cannot record this submission, "
                "so it is refused until Mongo recovers (a request that cannot "
                "be recorded is worse than a request that is refused)")
    return ""


def _check_budget(budget_state, action, params, risk):
    """"" or a reason to escalate an otherwise-direct decision to approval.

    Only ever escalates, never refuses outright and never touches a decision
    that was already "propose" or "refuse" — a request already headed for
    human sign-off does not need a second reason, and a request this module
    has already refused does not become more refused. Silent on anything it
    cannot evaluate (no budget_state, no declared remaining figure, or an
    action whose cost this table cannot estimate) rather than blocking a
    request over a number nobody supplied.
    """
    if risk not in ("R2", "R3") or not isinstance(budget_state, dict):
        return ""
    remaining = budget_state.get("su_remaining")
    if not isinstance(remaining, (int, float)) or isinstance(remaining, bool):
        return ""
    est = estimate_su(action, params)
    if est["su"] is None:
        return ""
    if est["su"] > remaining:
        return ("estimated %.3g SU exceeds the %.3g SU remaining in budget; "
                "escalated to approval rather than applied directly"
                % (est["su"], remaining))
    return ""


def authorize(actor, action, params=None, budget_state=None, resources=None):
    """{"allowed": bool, "needs_approval": bool, "risk": str|None, "reasons": [str]}.

    `risk` is the action's own risk tier when the action itself was resolved
    (even if the request is then refused on actor or params), and None only
    when the action id itself could not be resolved at all. Every reason is a
    complete English sentence naming the specific thing that decided the
    result, because a human — or a model reading its own refusal — has to be
    able to act on it without opening this file.

    Never raises. Every code path below either returns explicitly or falls
    into the catch-all at the bottom, which is the actual guarantee: no
    exception from a malformed table, a hostile params dict or a defect in
    this function itself can propagate as anything other than a refusal.
    """
    try:
        try:
            tier, _identity = _parse_actor(actor)
        except _PolicyDefect as exc:
            return _result(False, False, None, [str(exc)])

        try:
            action_id = _parse_action_id(action)
        except _PolicyDefect as exc:
            return _result(False, False, None, [str(exc)])

        table = _table()
        row = table["actions"].get(action_id)
        if row is None:
            hit = [e for e in table["errors"] if action_id in e]
            reason = ("action %r is not in the policy table" % action_id)
            if hit:
                reason += " (%s)" % hit[0]
            return _result(False, False, None, [reason])

        risk = row["risk"]

        ok, reasons = _check_params(params, row["param_bounds"])
        if not ok:
            return _result(False, False, risk, reasons)

        if tier not in row["allowed_tiers"]:
            return _result(False, False, risk, [
                "actor tier %r may not request action %r (allowed tiers: %s)"
                % (tier, action_id, ", ".join(sorted(row["allowed_tiers"])))])

        block = _check_resources(resources, risk)
        if block:
            return _result(False, False, risk, [block])

        decision = _decide(tier, risk)
        if decision == "refuse":
            if risk == "R4":
                return _result(False, False, risk, [
                    "%s is destructive/irreversible and human-only; refused, "
                    "not queued, for actor tier %r" % (risk, tier)])
            return _result(False, False, risk, [
                "actor tier %r has no authority at %s for action %r "
                "(ceiling: %s)" % (tier, risk, action_id,
                                   _CEILING.get(tier) or "no risk level granted")])

        needs_approval = (decision == "propose")
        reasons = []
        if needs_approval:
            reasons.append("%s at %s requires human approval before it may run"
                           % (tier, risk))
        else:
            reasons.append("%s is permitted to apply %s directly" % (tier, risk))
            escalate = _check_budget(budget_state, action_id, params, risk)
            if escalate:
                needs_approval = True
                reasons.append(escalate)

        return _result(True, needs_approval, risk, reasons)
    except Exception as exc:                       # actor/params/table are untrusted input
        return _result(False, False, None, [
            "authorize() raised %s (%s); refused rather than risk an "
            "unauthorized allow" % (type(exc).__name__, str(exc)[:200])])


# --- CLI ----------------------------------------------------------------------
def _dump(obj):
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="policy", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("table", help="the full loaded action table plus any load errors")

    exp = sub.add_parser("explain", help="one action's row, or its unknown reason")
    exp.add_argument("action")

    auth = sub.add_parser("authorize", help="run one authorization decision")
    auth.add_argument("actor")
    auth.add_argument("action")
    auth.add_argument("--params", default="{}", help="JSON object")
    auth.add_argument("--budget", default="null", help="JSON object or null")
    auth.add_argument("--resources", default="null", help="JSON object or null")

    args = ap.parse_args(argv)

    if args.cmd == "table":
        _dump({"tool_version": TOOL_VERSION, "path": _table()["path"],
              "errors": errors(), "actions": _table()["actions"]})
        return 0
    if args.cmd == "explain":
        _dump(describe(args.action))
        return 0
    if args.cmd == "authorize":
        try:
            params = json.loads(args.params)
            budget = json.loads(args.budget)
            resources = json.loads(args.resources)
        except Exception as exc:
            print("bad JSON argument: %s" % exc)
            return 2
        result = authorize(args.actor, args.action, params, budget, resources)
        _dump(result)
        return 0 if result["allowed"] else 1
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
