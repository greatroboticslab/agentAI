"""INC autopilot lever menu: diagnoses -> exact builder commands
(docs/INC_AUTOPILOT.md, (a) component 3 and (b), "Lever catalogue").

    propose(diagnoses, evidence) -> {"proposals", "cards", "operations", "deferred", "refused"}
    argv(lever, params, derived=None) -> [str, ...]

levers.json holds the menu: per lever its argv template, param_bounds (in
brain/policy_actions.json's form; checked with policy._check_params), policy
action, risk, control, success criterion, falsifier, predicted direction,
literature ids and cost estimator. levers.py renders a template into the
exact command a person runs by hand (tests/test_inc_ap_levers.py checks each
against its documented form and against the builder's own argparse), derives
what a proposer never supplies (L4's manifest paths from exp.json, child
experiment names), prices each build, and turns fired diagnoses into
model.proposal records (in-menu levers), model.card records (X1-X9, R4,
never queued) and operation records (advance, pause, halt, escalate: the
ticker's own moves).

Pricing (est_gpu_hours, the policy row's hours_param; 1 V100 GPU-hour = 1 SU):
  * pilot_rebuild (L1, L9): the parent pilot's cold arms (base, truth) and final
    runs cost what they cost (the definition is the same); each chain costs
    seeds x (cand + null images) x epochs x the seconds per image-epoch that
    chain measured in the parent (report gpu_hours / its train sizes). With
    full replay a step's pool is at most P0 plus every earlier clean
    increment, so the estimate is an upper bound.
  * realloop (L2, L5, L6): the same measured rates (latest complete pilot)
    applied to Step 1's base B size and the increment size M, over the
    realloop sequence (N verified + UNVERIFIED + OTHER_HEAVY).
  * baseline (L8): seeds x base images x cold epochs x the cold rate, plus
    one final run per seed.
  * walltime (L3, L4): the job script's own #SBATCH --time (one GPU).
  * zero (L7): the unit's runs were priced when the experiment was built.
  * every build (L1, L2, L5, L6, L8, L9) also pays for the job that builds it:
    run_inc_build.sh holds one V100 on GPU-shared for up to its #SBATCH
    --time (4 h, so at most 4 SU), added by price() (with_build_job) and
    recorded in the estimate as build_job_hours next to experiment_hours.
A build that cannot be priced from the evidence is deferred, never guessed.

price(lever, params, evidence, parent) is the pricing entry point: every
proposal here is priced through it. A price stated elsewhere (a brain plan's
est_gpu_hours) is never trusted; validate.materialise re-prices with the
same estimators, and price() is the call for any other filer.

The gate (docs/INCREMENTAL_PROTOCOL.md, Protocol v2): every lever that
rebuilds a pilot or a loop carries the pinned flips mode of the experiment it
follows (gate_params, from diagnose.pinned_gate): L1 its parent pilot's; L2
and L6 from D4 the mode of the pilot D4 evaluated; L2 from D1 and L5 the
parent loop's (loop_params). The flag --gate-flips-mode is rendered only
when that mode differs from the builders' default (levers.json protocol
gate_flips_mode_default, checked against driver.DEFAULT_FLIPS_MODE), so a v1
parent's command is byte for byte the one a person ran before the flag
existed. L9 (from D15) rebuilds a v1 pilot with its own replay mode and
--gate-flips-mode net: the pre-registered test of v2; it is never proposed
on a parent already pinned to net (D15 gives card X6 there). A parent whose
own final v2_check is refuted (D16) is rebuilt by no lever: gate_params
defers, since v2 is retired for that gate and card X7 leaves the choice
(v1 again, or a new guard) to a person.

Lever sequencing: a diagnosis lists what to do now in 'levers'; what comes
after it (D2: L2 with --relevance after L3) is in its detail 'then' and is
reported under 'deferred', unless the same tick already proposes it (a
proposal of that lever whose argv carries the flags the 'then' names:
D2's 'L2 with --increment-sources evidence' while D4's L2 does). The next
tick re-diagnoses and proposes it when its prerequisite holds (for L2:
relevance.json matching the select build, or the evidence criterion).

The relevance criterion of a real loop (docs/INCREMENTAL_PROTOCOL.md,
Steps 2-3; increment_criterion). realloop build draws its verified and
OTHER_HEAVY increments by one of two pre-registered criteria
(--increment-sources): the relevance file (the default, --relevance
INC_DIR/step1/relevance.json) or source-level species evidence
('evidence'). L2 and L6 from D4 take --relevance when step1/relevance.json
is made for the select build and passed its calibration check. When that
file is made for this select build but failed its own calibration check
(relevance_status 'calibration_failed': relevance.load refuses it, so every
--relevance build does), they take --increment-sources evidence, provided
the evidenced pool (evidence_capacity: the increment-pool images of sources
with verified cwd12-species boxes, from admit_summary.json and
select_summary.json) holds (N + 1) x M images. When it cannot hold the
builders' default N and M, the loop is sized by the R4 rule of 2026-09-27
(evidence_sizing): N = the fewest verified increments that decide
protocol min_decided_increments (6) with UNVERIFIED and OTHER_HEAVY, so 4;
M = min(the default M, floor(evidenced images / (N + 1))); the sized
--n-verified and --size are rendered where they differ from the defaults.
When that M is below protocol min_increment_frac (5 %) of base B, or the
capacity cannot be computed, they defer and choosing N and M is a
person's decision (D2 escalates). A build that gives its own N or M (a
brain plan's, an L5 or D1 rebuild's parent values) is never sized, but it
is held to the same floor (r4_floor: M at least min_increment_frac x |B|
and at least min_decided_increments decided increments), else refused
(LeverError), and then to the capacity. A missing file defers to L3; a
stale or malformed one (another format, made under another rule, or a
check that does not follow from its tau: relevance_status 'refused') to a
person. The flag is rendered only for
'evidence', in the slot --relevance takes, and never together with it; D2
records which criterion the build uses and why. L2 from D1 and L5 keep the
parent loop's own criterion (loop_params), checked again before the rebuild
(check_criterion). The OtherPlant-heavy part of realloop's capacity check
needs select_clusters.csv and is left to the builder.

Lineage: a lever is applied to a parent once. Diagnoses are re-run every
tick, and D4 is campaign-wide (it reads the latest complete pilot whatever
experiment is being looked at), so the same diagnosis keeps firing after its
lever ran. Before a build is proposed, applied() looks for its earlier
application, in the evidence and in the ticker's context 'lineage' records
(evidence.py): a full-replay pilot initialised after the parent (L1); a real
loop with the replay mode and recipes of D4's decision on the same base (L2,
L6 from D4); a real loop initialised after the parent loop with its recipes
and full replay (L2 from D1) or a larger size (L5); base_b_v1 (L8); a pilot
initialised after the parent and pinned to the gate L9 sets, net (L9). An
experiment counts only when it is decided on the gate the lever forwards
(gate_mode: the parent's for L1, L2 from D1 and L5; the evaluated pilot's
for L2 and L6 from D4), so a v1 loop never stands for a net L2 or the
reverse. Found, the lever is deferred naming that experiment. child_name
and realloop_name then only step past names taken by other experiments,
never past an earlier application of the same lever.
"""
from __future__ import annotations

import json
import posixpath
import re
from pathlib import Path

from . import evidence as E
from . import model as M

MENU_FILE = Path(__file__).resolve().parent / "levers.json"
# The bytes of this module as imported (diagnose.rules_files): D2's decision
# and the L2 it names rest on rules that live here (evidence_capacity, the R4
# sizing rule and its floor), so they are part of the rules version.
_SELF_BYTES = Path(__file__).resolve().read_bytes()
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
REALLOOP_STEM = "realloop_v"
BASELINE_B_EXP = "base_b_v1"                  # docs/INCREMENTAL_PROTOCOL.md Step 1.6
RECIPE_ORDER = ("full", "freeze", "lora")     # pilot.inc_recipes() order
KIND_VERIFIED = "verified"                    # realloop.KIND_VERIFIED (a step's "kind" in exp.json)
BUILD_SCRIPT = "run_inc_build.sh"             # the job every inc_build_* action submits (build_job_hours)
LINEAGE_IGNORED = ("failed", "refused", "cancelled")   # context lineage records that did not apply a lever
_SUB = re.compile(r"\{([a-z0-9_]+)\}")
_SPREAD = re.compile(r"\{\*([a-z0-9_]+)\}\Z")
_TIME = re.compile(r"^#SBATCH\s+--time=(?:(\d+)-)?(\d+):(\d+)(?::(\d+))?\s*$", re.M)


class LeverError(ValueError):
    """A lever that cannot be rendered: bad params or a missing value."""


class Defer(Exception):
    """A lever whose prerequisites are not in the evidence yet."""


# ------------------------------------------------------------------ menu
_CACHE = {}


def load_menu(path=None):
    """levers.json (cached on path and mtime)."""
    p = Path(path or MENU_FILE)
    key = (str(p), p.stat().st_mtime_ns)
    if key not in _CACHE:
        with open(p) as fh:
            _CACHE.clear()
            _CACHE[key] = json.load(fh)
    return json.loads(json.dumps(_CACHE[key]))


def row(lid, menu=None):
    menu = menu or load_menu()
    for blk in ("levers", "operations", "cards"):
        if lid in menu.get(blk, {}):
            return menu[blk][lid]
    raise LeverError("%r is not on the lever menu" % (lid,))


def protocol(key, menu=None):
    return (menu or load_menu())["protocol"][key]["value"]


def match_refusal(message, menu=None):
    """The levers.json refusal entry whose pattern the builder's message
    matches (first match), or None: the prerequisite of a refused build."""
    for r in (menu or load_menu()).get("refusals") or []:
        if re.search(r["pattern"], str(message or "")):
            return dict(r)
    return None


# ---------------------------------------------------------------- render
# Lineage keys of funnel steps that the autopilot derives from the evidence and
# that no planner sets: they are declared here, not in levers.json's
# param_bounds, which the planner's menu shows (brain_plan.menu_section). They
# are no flag of the command; the executor keeps them as meta params and the
# campaign's lineage keys a step by them (campaign.FUNNEL_KEY_PARAMS).
LINEAGE_BOUNDS = {
    "L10": {"detector": {"type": "enum", "value_type": "int", "values": [2],
                         "why": "the copy detector version the pre-registration requires (its amendment A2, "
                                "docs/FUNNEL_AUDIT.md 14; context funnel_leak, funnel_steps): the leak step of "
                                "version 2 writes funnel/leak_v2.json; the leak verb reads the version from the "
                                "pre-registration"}}}


def check_params(lid, params, menu=None):
    """(ok, reasons): params against the lever's param_bounds (and its
    LINEAGE_BOUNDS), with the policy gate's own checker (no coercion; an
    undeclared key is refused), and the one pair the bounds cannot express:
    --increment-sources evidence with --relevance, which realloop refuses
    (M.EVIDENCE_WITH_RELEVANCE)."""
    from ..brain import policy
    bounds = row(lid, menu).get("param_bounds")
    if not isinstance(bounds, dict):
        return False, ["%s declares no param_bounds" % lid]
    bounds = dict(bounds, **LINEAGE_BOUNDS.get(lid, {}))
    ok, reasons = policy._check_params(params, bounds)
    if evidence_with_relevance(params):
        ok, reasons = False, list(reasons) + [M.EVIDENCE_WITH_RELEVANCE]
    return ok, reasons


def evidence_with_relevance(params):
    """True when params ask for --increment-sources evidence and name a
    --relevance file too (realloop build refuses the two together)."""
    p = params if isinstance(params, dict) else {}
    return p.get("increment_sources") == SOURCES_EVIDENCE and p.get("relevance") not in (None, "")


def render(template, values):
    """Tokens of an argv template (levers.json _meta.argv_template)."""
    out = []
    for t in template:
        if isinstance(t, dict):
            if values.get(t["if"]) is not None:
                out.extend(render(t["tokens"], values))
            continue
        m = _SPREAD.match(t)
        if m:
            v = values.get(m.group(1))
            if not isinstance(v, (list, tuple)) or not v:
                raise LeverError("%s needs a non-empty list" % t)
            out.extend(str(x) for x in v)
            continue

        def sub(mm):
            if values.get(mm.group(1)) is None:
                raise LeverError("no value for {%s}" % mm.group(1))
            return str(values[mm.group(1)])
        out.append(_SUB.sub(sub, t))
    return out


def argv(lid, params, derived=None, menu=None):
    """The exact command of lever lid: params (checked against its bounds,
    fixed values included) and derived values (read from the evidence)."""
    r = row(lid, menu)
    if not r.get("argv"):
        raise LeverError("%s has no command (a card or a campaign operation)" % lid)
    ok, reasons = check_params(lid, params, menu)
    if not ok:
        raise LeverError("%s params refused: %s" % (lid, "; ".join(reasons)))
    for k, v in (r.get("fixed") or {}).items():
        if k in params and params[k] != v:
            raise LeverError("%s fixes %s = %r" % (lid, k, v))
    missing = [k for k in r.get("requires") or [] if params.get(k) is None]
    if missing:
        raise LeverError("%s requires %s" % (lid, missing))
    values = dict(derived or {})
    values.update(params)
    return render(r["argv"], values)


# ----------------------------------------------------------------- names
def child_name(parent, existing):
    """The next experiment after `parent`: <stem>_v<n+1> for <stem>_v<n>
    (pilot_v1 -> pilot_v2, the manual command), else <parent>_rf; never a
    name in `existing` (an experiment is built once). It steps past names
    other experiments hold; whether the lever was already applied to parent
    is applied()'s question, asked before any name is chosen."""
    existing = set(existing or ())
    m = re.fullmatch(r"(.+)_v(\d+)", parent)
    if m:
        n = int(m.group(2)) + 1
        while "%s_v%d" % (m.group(1), n) in existing:
            n += 1
        name = "%s_v%d" % (m.group(1), n)
    else:
        name, i = parent + "_rf", 2
        while name in existing:
            name, i = "%s_rf%d" % (parent, i), i + 1
    if not NAME_RE.match(name):
        raise LeverError("child name %r is not a valid experiment name" % name)
    return name


def realloop_name(existing):
    """realloop_v<n>, the first n not in `existing` (see child_name)."""
    n = 1
    while "%s%d" % (REALLOOP_STEM, n) in set(existing or ()):
        n += 1
    return "%s%d" % (REALLOOP_STEM, n)


def inc_dir(ev, exp=None):
    """The cluster INC_DIR the experiment lives in (its exp.json base
    manifest is INC_DIR/<exp>/manifests/<name>.jsonl), else model's."""
    for e in ([exp] if exp else []) + list(reversed(ev.exps())):
        m = ((ev.json("%s/exp.json" % e) or {}).get("base") or {}).get("manifest")
        if isinstance(m, str) and m.startswith("/") and posixpath.basename(posixpath.dirname(m)) == "manifests":
            return posixpath.dirname(posixpath.dirname(posixpath.dirname(m)))
    return M.CLUSTER_INC_DIR


# ----------------------------------------------------------------- derive
def audit_args(ev, exp):
    """L4's derived values from exp.json: trusted = the base manifest; the
    audited steps as NAME=manifest, clean steps in sequence order, then the
    others (run_inc_audit.sh's documented pilot command); out =
    INC_DIR/<exp>/audit/label_audit.json."""
    d = ev.json("%s/exp.json" % exp)
    if d is None:
        raise Defer("no %s/exp.json in the evidence" % exp)
    steps = d.get("steps") or []
    order = [s for s in steps if s.get("clean")] + [s for s in steps if not s.get("clean")]
    audits = ["%s=%s" % (s["name"], s["manifest"]) for s in order]
    cites = [ev.cite("%s/exp.json" % exp, "/base/manifest")]
    cites += [ev.cite("%s/exp.json" % exp, "/steps/%d/manifest" % steps.index(s)) for s in order]
    return {"trusted": d["base"]["manifest"], "audits": audits,
            "out": posixpath.join(inc_dir(ev, exp), exp, "audit", "label_audit.json")}, cites


def _recipes_param(names):
    names = [n for n in RECIPE_ORDER if n in names] + sorted(n for n in names if n not in RECIPE_ORDER)
    return ",".join(names)


RELEVANCE_TABLES = (("increment_pool", "increment_pool.jsonl"), ("base_selected", "base_selected.jsonl"))
SEL, ADMIT, REL = "step1/select_summary.json", "step1/admit_summary.json", "step1/relevance.json"
SOURCES_RELEVANCE, SOURCES_EVIDENCE = "relevance", "evidence"   # select.SOURCES_* (protocol increment_sources_modes)
CALIBRATION_FAILED = "calibration_failed"
SOURCES_RECOVERED = "recovered"                # realloop.SOURCES_RECOVERED (the funnel's overlay, realloop_v2)


def relevance_status(ev, menu=None):
    """{"state", "why", "tau", "tau_min"}: whether step1/relevance.json is one
    relevance.load would accept for the select build in the evidence, and
    when not, why. The file is read in relevance.load's order (format, the
    rule it was made under, tau, the calibration check), then compared with
    the select build. state is one of:
      * missing: no file;
      * refused: relevance.load's cheap checks refuse it other than by a
        clean calibration failure: not an inc.relevance/1 file; made under
        another rule (params min_crops, calibration_percentile, tau_min not
        the protocol's relevance_min_crops, relevance_calibration_percentile,
        relevance_tau_min); no calibrated tau in [0, 1]; no check block; or
        a check that does not follow from its tau under the rule (ok not a
        boolean, a check tau_min other than relevance_tau_min, or ok not
        equal to tau >= relevance_tau_min: ok true below it, ok false at or
        above it);
      * stale: made for another select build (each table's manifest sha256
        as select_summary.json records it; a table the select summary does
        not record is not compared). A stale file is stale whatever its
        check says;
      * calibration_failed: made for this select build under the protocol's
        rule, and its own calibration check failed (ok false, tau <
        tau_min = relevance_tau_min): relevance.load refuses it, every
        --relevance build refuses it, and the protocol's second criterion,
        source-level species evidence, applies (increment_criterion);
      * matching: made for this select build under the protocol's rule with
        a passed check (ok true, tau >= tau_min = relevance_tau_min).
    The builder's full check (every source status re-derived from its
    numbers) runs on the cluster."""
    sel, rel = ev.json(SEL), ev.json(REL)
    if rel is None:
        return {"state": "missing", "why": "no step1/relevance.json", "tau": None, "tau_min": None}
    cal = rel.get("calibration") if isinstance(rel.get("calibration"), dict) else {}
    tau = _num(cal.get("tau"))
    chk = cal.get("check") if isinstance(cal.get("check"), dict) else None
    tmin = _num((chk or {}).get("tau_min"))
    out = {"tau": tau, "tau_min": tmin}
    if rel.get("format") != protocol("relevance_format", menu):
        return dict(out, state="refused", why="format %r, not %s" % (rel.get("format"),
                                                                     protocol("relevance_format", menu)))
    want = protocol("relevance_tau_min", menu)
    rule = (protocol("relevance_min_crops", menu), protocol("relevance_calibration_percentile", menu), want)
    prm = rel.get("params") if isinstance(rel.get("params"), dict) else {}
    got = (prm.get("min_crops"), prm.get("calibration_percentile"), prm.get("tau_min"))
    # relevance.load compares with != (so 20.0 is 20); a boolean is never a rule value
    if any(isinstance(g, bool) or g != r for g, r in zip(got, rule)):
        return dict(out, state="refused",
                    why="made under another rule (params min_crops %r, calibration_percentile %r, tau_min %r; "
                        "the protocol: %r, %r, %r)" % (got + rule))
    if tau is None or not 0.0 <= tau <= 1.0:
        return dict(out, state="refused", why="no calibrated tau in [0, 1]")
    if chk is None:
        return dict(out, state="refused", why="no calibration check")
    ok = chk.get("ok")
    if not isinstance(ok, bool) or tmin is None or tmin != want or ok != (tau >= want):
        return dict(out, state="refused",
                    why="a calibration check that does not follow from its tau under the rule (ok %r, tau %r, "
                        "tau_min %r; the rule: tau >= %g)" % (ok, tau, chk.get("tau_min"), want))
    outs = (sel or {}).get("outputs") or {}
    compared = 0
    for table, fname in RELEVANCE_TABLES:
        wsha = (outs.get(fname) or {}).get("sha256")
        if wsha is None and table != "increment_pool":
            continue
        gsha = (((rel.get("inputs") or {}).get(table)) or {}).get("sha256")
        if not wsha or gsha != wsha:
            return dict(out, state="stale", why="made for another %s (%s) than the select build's (%s)"
                        % (fname, str(gsha)[:12], str(wsha)[:12]))
        compared += 1
    if not compared:
        return dict(out, state="stale", why="the select build records no increment_pool.jsonl to compare with")
    if not ok:
        return dict(out, state=CALIBRATION_FAILED,
                    why="its calibration check failed: tau %s < %s (relevance.load refuses it)" % (tau, tmin))
    return dict(out, state="matching", why="made for this select build; calibration check passed")


def relevance_state(ev, menu=None):
    """relevance_status(ev)["state"]: 'missing', 'refused', 'stale',
    'calibration_failed' or 'matching'."""
    return relevance_status(ev, menu)["state"]


def relevance_cites(ev):
    """Cites of step1/relevance.json's format and calibration check (those it holds)."""
    out = []
    for ptr in ("/format", "/calibration/tau", "/calibration/check/ok", "/calibration/check/tau_min"):
        try:
            out.append(ev.cite(REL, ptr))
        except KeyError:
            pass
    return out


RELEVANCE_WHY = {
    "missing": "step1/relevance.json is missing: a production realloop build refuses without one made for this "
               "select build (L3 first)",
    "stale": "step1/relevance.json is stale (made for another select build): relevance build refuses over it "
             "without --force, which is not on the menu; a person decides",
    "refused": "step1/relevance.json is refused (not an inc.relevance/1 file made under the protocol's rule with a "
               "calibration check that follows from its tau): every realloop build refuses it; a person decides",
    CALIBRATION_FAILED: "step1/relevance.json failed its own calibration check: relevance.load refuses it, so no "
                        "--relevance build runs; a real loop from D4's decision uses source-level species evidence "
                        "(--increment-sources evidence) instead, and a rebuild that names the file is a person's "
                        "decision",
}
EVIDENCE_WHY = ("step1/relevance.json is made for this select build but failed its own calibration check, so "
                "relevance.load refuses it and no --relevance build runs; the protocol's second criterion, "
                "source-level species evidence (docs/INCREMENTAL_PROTOCOL.md, Steps 2-3; adopted at the R4 review "
                "of 2026-09-27), rests on the verifier instead: a source is drawn from only if verify admit judged "
                "at least min_evidence of its cwd12-species boxes verified")
OTHER_HEAVY_UNCHECKED = ("not checked here: the OtherPlant-heavy share of the evidenced pool needs select_clusters.csv, "
                         "which is never shipped; realloop build checks it (select.pool_capacity) before anything is "
                         "written, and a refusal comes back as a builder refusal")


def _relevance_path_ok(ev, path, inc):
    """Defer unless `path` (a build's --relevance) is usable: the Step 1 file
    must match the select build in the evidence; another path is the
    builder's to check."""
    if path == posixpath.join(inc, "step1", "relevance.json"):
        state = relevance_state(ev)
        if state != "matching":
            raise Defer(RELEVANCE_WHY[state])


def _box_count(x):
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def evidence_capacity(ev, n_verified=None, size=None, menu=None):
    """(record, cites): the source-level species evidence criterion (realloop
    build --increment-sources evidence; docs/INCREMENTAL_PROTOCOL.md, Steps
    2-3) on the Step 1 summaries in the evidence, the way select.load_evidence,
    select.apply_evidence and realloop.evidence_capacity apply it:
      * per source, the cwd12-species boxes verify admit judged 'verified'
        (admit_summary.json per_slug[source].boxes.verified, 0 when absent);
        a source with at least min_evidence (protocol min_evidence_default;
        --min-evidence is not on the menu) is evidenced;
      * the evidenced sources' increment-pool images (select_summary.json
        sources.increment_pool) are what the verified and OTHER_HEAVY draws
        may take;
      * fits: they hold (N + 1) x M images, N the verified increments
        (n_verified, else protocol n_verified_default) and M the increment
        size (size, else inc_frac of base B's images, realloop.default_size);
        max_n_verified is the largest N the image count allows at this M.
    The OtherPlant-heavy part of realloop's capacity check (M of those images
    in OtherPlant-heavy near-dup groups) needs select_clusters.csv, which is
    never shipped: it is recorded as not checked here (other_heavy), and the
    builder checks it before anything is written.
    Raises Defer when the summaries are not in the evidence, or disagree in a
    way the builder refuses: an admit summary without per_slug verdict
    counts, one that names another verified.jsonl or crops.csv than the
    select build read, a source whose verified boxes exceed its species
    crops in select_summary.json retrieval.source_evidence, a source that
    record counts without a per_slug entry, or an increment-pool source
    without one."""
    sel, admit = ev.json(SEL), ev.json(ADMIT)
    if sel is None or admit is None:
        raise Defer("source-level species evidence reads step1/select_summary.json and step1/admit_summary.json; "
                    "%s is not in the evidence" % (SEL if sel is None else ADMIT))
    min_ev = protocol("min_evidence_default", menu)
    per_slug = admit.get("per_slug")
    if not isinstance(per_slug, dict) or not per_slug:
        raise Defer("step1/admit_summary.json holds no per_slug verdict counts: realloop build --increment-sources "
                    "evidence refuses it (rerun inc.verify admit)")
    cites = []
    inputs = sel.get("inputs") or {}
    for name, field in (("verified", "verified_sha256"), ("crops", "crops_sha256")):
        want = (inputs.get(name) or {}).get("sha256")
        if not want or admit.get(field) != want:
            raise Defer("step1/admit_summary.json and select_summary.json disagree: the admit summary names another "
                        "%s (%s) than the select build read (%s); realloop build --increment-sources evidence refuses "
                        "(rerun select build after verify admit)"
                        % ("verified.jsonl" if name == "verified" else "crops.csv", str(admit.get(field))[:12],
                           str(want)[:12]))
        cites += [ev.cite(ADMIT, "/" + field), ev.cite(SEL, E.pointer("inputs", name, "sha256"))]
    verified = {}
    for s, d in sorted(per_slug.items()):
        boxes = d.get("boxes") if isinstance(d, dict) else None
        v = (boxes or {}).get("verified", 0)
        if not isinstance(boxes, dict) or not _box_count(v):
            raise Defer("step1/admit_summary.json per_slug[%r] has no box verdict counts: realloop build "
                        "--increment-sources evidence refuses it" % s)
        verified[s] = v
    se = (sel.get("retrieval") or {}).get("source_evidence")
    if se is None:
        cross = {"checked": False, "reason": "select_summary.json records no retrieval.source_evidence"}
    else:
        if not isinstance(se, dict):
            raise Defer("select_summary.json retrieval.source_evidence is not a table: realloop refuses")
        bad = []
        for s in sorted(set(verified) | set(se)):
            n_sp = (se.get(s) or {}).get("species_crops", 0) if isinstance(se.get(s) or {}, dict) else None
            if not _box_count(n_sp):
                bad.append("%s: species_crops %r is not a count" % (s, n_sp))
            elif s not in verified:
                bad.append("%s: %d species crop(s), no per_slug entry" % (s, n_sp))
            elif verified[s] > n_sp:
                bad.append("%s: %d verified box(es), %d species crop(s)" % (s, verified[s], n_sp))
        if bad:
            raise Defer("step1/admit_summary.json and select_summary.json disagree for %d source(s) (every source "
                        "needs species_crops >= verified boxes): %s; realloop build --increment-sources evidence "
                        "refuses" % (len(bad), "; ".join(bad[:5])))
        cross = {"checked": True, "sources_compared": len(set(verified) | set(se))}
    pool = (sel.get("sources") or {}).get("increment_pool")
    if not isinstance(pool, dict) or not pool or not all(_box_count(n) for n in pool.values()):
        raise Defer("select_summary.json records no per-source increment-pool image counts "
                    "(sources.increment_pool) to size the evidenced pool")
    unknown = sorted(s for s in pool if s not in verified)
    if unknown:
        raise Defer("%d increment-pool source(s) have no per_slug entry in step1/admit_summary.json, e.g. %s: "
                    "realloop build --increment-sources evidence refuses" % (len(unknown), unknown[:3]))
    n_base = _num((sel.get("sizes") or {}).get("base_B"))
    if not n_base:
        raise Defer("no Step 1 select build (sizes.base_B) in the evidence to size the increments")
    evidenced = {s: n for s, n in verified.items() if n >= min_ev}
    held = {s: int(pool[s]) for s in sorted(pool) if s in evidenced and pool[s]}
    excluded = {s: int(n) for s, n in sorted(pool.items()) if s not in evidenced and n}
    m = int(size) if size is not None else max(1, int(round(protocol("inc_frac", menu) * n_base)))
    nv = int(n_verified) if n_verified is not None else int(protocol("n_verified_default", menu))
    images = sum(held.values())
    need = (nv + 1) * m
    rec = {"mode": SOURCES_EVIDENCE, "min_evidence": min_ev,
           "evidenced_sources": {s: {"verified_boxes": evidenced[s], "pool_images": int(pool.get(s, 0))}
                                 for s in sorted(evidenced)},
           "evidenced_pool_images": images, "increment_pool_images": int(sum(pool.values())),
           "excluded_sources": len(excluded), "excluded_images": int(sum(excluded.values())),
           "base_images": int(n_base), "n_verified": nv, "increment_images": m, "needed_images": need,
           "fits": images >= need, "max_n_verified": images // m - 1,
           "other_heavy": OTHER_HEAVY_UNCHECKED, "cross_check": cross}
    cites.append(ev.cite(SEL, "/sizes/base_B"))
    for s in sorted(evidenced):
        cites.append(ev.cite(ADMIT, E.pointer("per_slug", s, "boxes", "verified")))
        if s in pool:
            cites.append(ev.cite(SEL, E.pointer("sources", "increment_pool", s)))
    return rec, cites


def capacity_why(rec):
    """Why the evidenced pool cannot supply a build (evidence_capacity's or
    evidence_sizing's record; with the latter, also why the R4 sizing rule
    did not size it)."""
    sz = rec.get("sizing") or {}
    rule = ("; %s" % sizing_why(rec)) if sz.get("sized") and not sz.get("applied") else ""
    return ("source-level species evidence cannot supply this build: %d evidenced source(s) (at least %d verified "
            "cwd12-species box(es)) hold %d of the %d increment-pool images, and N = %d verified increments + "
            "OTHER_HEAVY of M = %d images need %d (at this M the image count allows at most --n-verified %d)%s; "
            "choosing N and M under this criterion is then a review decision (docs/INCREMENTAL_PROTOCOL.md, Steps "
            "2-3): a person"
            % (len([s for s, e in rec["evidenced_sources"].items() if e["pool_images"]]), rec["min_evidence"],
               rec["evidenced_pool_images"], rec["increment_pool_images"], rec["n_verified"], rec["increment_images"],
               rec["needed_images"], rec["max_n_verified"], rule))


# The sizing rule of an evidence-sourced real loop (the R4 review of 2026-09-27;
# docs/INC_AUTOPILOT.md D2 and docs/INCREMENTAL_PROTOCOL.md Steps 2-3).
SIZING_RULE = ("R4 review of 2026-09-27: an evidence-sourced real loop whose default N and M do not fit the evidenced "
               "pool is sized by rule instead of escalated. N = the fewest verified increments with which the loop "
               "decides at least protocol.min_decided_increments increments together with UNVERIFIED and "
               "OTHER_HEAVY; M = min(round(inc_frac x |B|), floor(evidenced pool images / (N + 1))). An M below "
               "min_increment_frac x |B| (increments too small for the gate to detect an effect), or a capacity "
               "that cannot be computed, is not sized: D2 escalates with the numbers. The OtherPlant-heavy share "
               "stays realloop build's check.")
SIZED_KEYS = ("n_verified", "increment_images", "needed_images", "fits", "max_n_verified")


def sized_n_verified(menu=None):
    """N of the sizing rule: the fewest verified increments (within L2's
    n_verified bound) whose realloop sequence, with UNVERIFIED and
    OTHER_HEAVY, holds protocol min_decided_increments steps (4 for 6)."""
    want = int(protocol("min_decided_increments", menu))
    hi = int(row("L2", menu)["param_bounds"]["n_verified"]["max"])
    for n in range(1, hi + 1):
        if len(realloop_sequence(n)) >= want:
            return n
    raise Defer("no --n-verified within L2's bound (%d) decides %d increments" % (hi, want))


def evidence_sizing(ev, menu=None):
    """(record, cites): evidence_capacity at the builders' default N and M;
    when those do not fit, at N and M sized by the R4 rule (SIZING_RULE):
    N = sized_n_verified(), M = min(the default M, floor(evidenced pool
    images / (N + 1))). The record is evidence_capacity's at the N and M the
    build would use, plus 'sizing':
      rule, min_decided_increments, min_increment_frac, min_increment_images
      (min_increment_frac x |B|); default {n_verified, increment_images,
      needed_images, fits, max_n_verified, decided_increments}; sized (None
      when the default fits) {n_verified, increment_images, needed_images,
      decided_increments, increment_frac, fits}; applied (the build uses the
      sized N and M); params and flags (the non-default --size and
      --n-verified, in the L2 template's order); why.
    When the sized M is below min_increment_frac x |B| (or below 1 image) the
    loop is not sized: the record stays the default one (fits False) with
    the refused sizing recorded, and the caller escalates. Raises Defer when
    the capacity cannot be computed (evidence_capacity)."""
    rec, cites = evidence_capacity(ev, menu=menu)
    default = dict({k: rec[k] for k in SIZED_KEYS}, decided_increments=len(realloop_sequence(rec["n_verified"])))
    n_base = rec["base_images"]
    frac = protocol("min_increment_frac", menu)
    floor_images = frac * n_base
    sizing = {"rule": SIZING_RULE, "min_decided_increments": protocol("min_decided_increments", menu),
              "min_increment_frac": frac, "min_increment_images": round(floor_images, 2), "default": default,
              "sized": None, "applied": False, "params": {}, "flags": []}
    if rec["fits"]:
        sizing["why"] = "the default build fits the evidenced pool: not sized"
        return dict(rec, sizing=sizing), cites
    nv = sized_n_verified(menu)
    m = min(rec["increment_images"], rec["evidenced_pool_images"] // (nv + 1))
    sized = {"n_verified": nv, "increment_images": m, "needed_images": (nv + 1) * m,
             "decided_increments": len(realloop_sequence(nv)), "increment_frac": round(m / float(n_base), 4)}
    if m < 1 or m < floor_images:
        sizing["sized"] = dict(sized, fits=False)
        sizing["why"] = "not sized: M = %d is below min_increment_frac x |B| = %s" % (m, round(floor_images, 2))
        return dict(rec, sizing=sizing), cites
    srec, scites = evidence_capacity(ev, nv, m, menu)
    params = {}
    if m != default["increment_images"]:
        params["size"] = m
    if nv != default["n_verified"]:
        params["n_verified"] = nv
    tmpl = [t for t in row("L2", menu)["argv"] if isinstance(t, dict) and t.get("if") in ("size", "n_verified")]
    sizing.update(sized=dict(sized, fits=srec["fits"]), applied=srec["fits"], params=params,
                  flags=render(tmpl, params),
                  why="the default N = %d x M = %d needs %d images and the evidenced pool holds %d: sized by the R4 "
                      "rule to N = %d (%d decided increments) and M = min(%d, floor(%d / %d)) = %d (%.1f%% of base "
                      "B's %d images, at least the %g floor), needing %d"
                      % (default["n_verified"], default["increment_images"], default["needed_images"],
                         rec["evidenced_pool_images"], nv, sized["decided_increments"], default["increment_images"],
                         rec["evidenced_pool_images"], nv + 1, m, 100.0 * m / n_base, n_base, frac,
                         sized["needed_images"]))
    return dict(srec, sizing=sizing), scites


def sizing_why(rec):
    """Why the R4 sizing rule did not size a loop (evidence_sizing's record
    whose sizing was refused)."""
    sz = rec.get("sizing") or {}
    s = sz.get("sized") or {}
    return ("the R4 sizing rule gives N = %s (%s decided increments with UNVERIFIED and OTHER_HEAVY) and M = min(%s, "
            "floor(%s / %s)) = %s images, below min_increment_frac %s x base B's %s images = %s: the R4 review "
            "judged increments that small too small for the gate to detect an effect, so the loop is not sized"
            % (s.get("n_verified"), s.get("decided_increments"), (sz.get("default") or {}).get("increment_images"),
               rec.get("evidenced_pool_images"), (s.get("n_verified") or 0) + 1, s.get("increment_images"),
               sz.get("min_increment_frac"), rec.get("base_images"), sz.get("min_increment_images")))


def r4_floor(rec, menu=None):
    """"" when an evidence-sourced real loop at the record's N and M
    (evidence_capacity's n_verified, increment_images and base_images) is at
    or above the R4 rule's floor, else why not: M below protocol
    min_increment_frac x |B|, or fewer than protocol min_decided_increments
    decided increments (N verified with UNVERIFIED and OTHER_HEAVY,
    realloop_sequence). evidence_sizing never sizes a loop below it; a
    build that gives its own N or M is held to it here."""
    n_base, nv, m = rec["base_images"], rec["n_verified"], rec["increment_images"]
    frac = protocol("min_increment_frac", menu)
    want = int(protocol("min_decided_increments", menu))
    floor_images = frac * n_base
    decided = len(realloop_sequence(nv))
    bad = []
    if m < 1 or m < floor_images:
        bad.append("M = %d is %.2f%% of base B's %d images, below min_increment_frac %g x |B| = %s images"
                   % (m, 100.0 * m / n_base, n_base, frac, round(floor_images, 2)))
    if decided < want:
        bad.append("N = %d decides %d increments with UNVERIFIED and OTHER_HEAVY, fewer than "
                   "min_decided_increments %d" % (nv, decided, want))
    if not bad:
        return ""
    return ("--n-verified %d --size %d is below the R4 rule's floor for an evidence-sourced real loop (R4 review of "
            "2026-09-27): %s. The rule sizes a loop at or above it (at least %d decided increments of at least %s "
            "images); the review judged smaller increments too small for the gate to detect an effect and fewer "
            "decided increments short of the Steps 2-3 acceptance, so a loop below the floor is a person's "
            "decision, built by hand, not a lever's (docs/INCREMENTAL_PROTOCOL.md, Steps 2-3)"
            % (nv, m, "; ".join(bad), want, round(floor_images, 2)))


def increment_criterion(ev, inc, n_verified=None, size=None, menu=None):
    """(flags, record, cites): the relevance criterion of a real-loop build
    from Step 1 (docs/INCREMENTAL_PROTOCOL.md, Steps 2-3), as the builder's
    flags:
      * {"relevance": INC_DIR/step1/relevance.json} when that file is made for
        the select build in the evidence and passed its calibration check
        (the builders' default criterion: --increment-sources is not
        rendered);
      * {"increment_sources": "evidence"} when that file is made for this
        select build but failed its own calibration check
        (relevance_status 'calibration_failed') and the evidenced pool holds
        the build's increments: source-level species evidence, the
        protocol's second criterion. --relevance is not rendered (realloop
        refuses the two together) and neither is --min-evidence (the
        protocol's default). With n_verified and size both None (a build
        from D4's decision) the capacity is evidence_sizing's: the default N
        and M when they fit, else N and M sized by the R4 rule, whose
        non-default values join the flags ({"size": M, "n_verified": N},
        rendered by the L2 template). A given n_verified or size (a brain
        plan's own, a rebuild's parent values) is never sized: it is held to
        the R4 rule's floor (r4_floor; below it LeverError, with the numbers)
        and then checked as given (evidence_capacity).
    Defers otherwise: no file (L3 first), a stale or malformed one (a person),
    and a calibration-failed file whose evidenced pool cannot be read or is
    too small even for the sized loop (a person decides N and M:
    capacity_why)."""
    st = relevance_status(ev, menu)
    if st["state"] == "matching":
        return ({"relevance": posixpath.join(inc, "step1", "relevance.json")},
                {"mode": SOURCES_RELEVANCE, "relevance": st}, [])
    if st["state"] != CALIBRATION_FAILED:
        raise Defer(RELEVANCE_WHY[st["state"]])
    if n_verified is None and size is None:
        rec, cites = evidence_sizing(ev, menu)
    else:
        rec, cites = evidence_capacity(ev, n_verified, size, menu)
        below = r4_floor(rec, menu)
        if below:
            raise LeverError(below)
    rec = dict(rec, relevance=st, why=EVIDENCE_WHY)
    if not rec["fits"]:
        raise Defer(capacity_why(rec))
    flags = {"increment_sources": SOURCES_EVIDENCE}
    if (rec.get("sizing") or {}).get("applied"):
        flags.update(rec["sizing"]["params"])           # the sized loop's non-default --size / --n-verified
    return flags, rec, relevance_cites(ev) + cites


def _l3_ok(ev):
    """Defer L3 (relevance build) over a relevance.json made for this select
    build whose calibration check failed: the build would only rewrite the
    same verdict (or refuse without --force), and the protocol's second
    criterion applies instead (D2, increment_criterion). Any proposer."""
    st = relevance_status(ev)
    if st["state"] == CALIBRATION_FAILED:
        raise Defer("L3 is not proposed over step1/relevance.json, which %s: a rebuild on the same select build "
                    "gives the same verdict; the source-level species evidence criterion applies (D2), and card "
                    "X9 is where a person revisits the zero-shot criterion" % st["why"])


def check_criterion(ev, params, inc):
    """[cites]; Defer unless the relevance criterion a rebuild of a real loop
    carries (loop_params: the parent's own) can build it now: a Step 1
    relevance.json it names must match the select build (_relevance_path_ok),
    and an evidence build must be at or above the R4 rule's floor at its own
    N and M (r4_floor; else LeverError) and its evidenced pool must hold its
    increments there (evidence_capacity)."""
    if params.get("increment_sources") == SOURCES_EVIDENCE:
        rec, cites = evidence_capacity(ev, params.get("n_verified"), params.get("size"))
        below = r4_floor(rec)
        if below:
            raise LeverError(below)
        if not rec["fits"]:
            raise Defer(capacity_why(rec))
        return cites
    _relevance_path_ok(ev, params.get("relevance"), inc)
    return []


def gate_mode(ev, exp, menu=None):
    """The flips mode experiment `exp` is decided with (diagnose.pinned_gate),
    the builders' default when no record names one."""
    from . import diagnose as DG
    mode = DG.pinned_gate(ev, exp)[0]
    return mode if mode is not None else protocol("gate_flips_mode_default", menu)


def gate_params(ev, exp, menu=None):
    """({"gate_flips_mode": mode} or {}, cites): the builders' --gate-flips-mode
    that carries experiment `exp`'s pinned flips mode (diagnose.pinned_gate).
    Empty when that mode is the builders' default (protocol
    gate_flips_mode_default) or no record names one (GateConfig's default):
    the builder then writes that mode without the flag. Defers on a mode no
    builder takes, and on a gate whose pre-registered check came out refuted
    on `exp` (report.json v2_check final 'refuted', D16): v2 is retired for
    that gate, so no lever rebuilds on it; card X7 leaves the choice (v1
    again, or a new guard) to a person."""
    from . import diagnose as DG
    mode, cite, _src = DG.pinned_gate(ev, exp)
    if mode is None or mode == protocol("gate_flips_mode_default", menu):
        return {}, []
    if mode not in protocol("gate_flips_modes", menu):
        raise Defer("%s is pinned to flips_mode %r, which no builder's --gate-flips-mode takes" % (exp, mode))
    chk, _cc = DG.v2_check_of(ev, exp)
    if chk is not None and chk.get("final") is True and chk.get("outcome") == "refuted":
        raise Defer("%s was decided on flips_mode %s and its protocol v2 check is refuted (D16): v2 is retired "
                    "for that gate, so no rebuild carries it; a person reverts to v1 or rethinks the guard (X7)"
                    % (exp, mode))
    return {"gate_flips_mode": mode}, ([cite] if cite is not None else [])


# ----------------------------------------------------------------- price
def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def rates(ev, exp):
    """Seconds per image-epoch measured in a finished chain experiment: per
    recipe (incremental runs) and cold (base + truth runs), and GPU-hours per
    final run. Raises Defer when the evidence cannot give them."""
    d, rep = ev.json("%s/exp.json" % exp), ev.json("%s/report.json" % exp)
    if d is None or rep is None:
        raise Defer("%s has no exp.json and report.json to measure its cost" % exp)
    gh = rep.get("gpu_hours") or {}
    seeds = len(d.get("seeds") or [])
    art = "%s/report.json" % exp
    cites, inc = [], {}
    for r, rec in sorted((d.get("recipes") or {}).items()):
        img = 0
        for st in rep.get("steps") or []:
            ti = (((st.get("chains") or {}).get(r) or {}).get("train_images")) or {}
            img += seeds * ((_num(ti.get("cand")) or 0) + (_num(ti.get("null")) or 0)) * rec["epochs"]
        h = _num((gh.get("chain:%s" % r) or {}).get("hours"))
        if img and h:
            inc[r] = h * 3600.0 / img
            cites.append(ev.cite(art, "/gpu_hours/%s/hours" % E._esc("chain:%s" % r)))
    cold_w = (d.get("effective_warmup") or {}).get("cold") or {}
    base_n = _num(((cold_w.get("base") or {}).get("images")))
    truth_n = sum(_num(v.get("images")) or 0 for v in (cold_w.get("truth") or {}).values())
    e_base = _num((d.get("base") or {}).get("recipe", {}).get("epochs"))
    e_truth = _num((d.get("truth_recipe") or {}).get("epochs")) or e_base
    h_cold = (_num((gh.get("base") or {}).get("hours")) or 0) + (_num((gh.get("truth") or {}).get("hours")) or 0)
    img_cold = seeds * ((base_n or 0) * (e_base or 0) + truth_n * (e_truth or 0))
    fin = gh.get("final") or {}
    if not inc or not img_cold or not h_cold or not fin.get("runs"):
        raise Defer("%s's report does not give per-image GPU time for every arm" % exp)
    cites += [ev.cite(art, "/gpu_hours/base/hours"), ev.cite(art, "/gpu_hours/truth/hours"),
              ev.cite(art, "/gpu_hours/final/hours")]
    return {"from": exp, "seeds": seeds, "inc_s_per_image_epoch": inc,
            "cold_s_per_image_epoch": h_cold * 3600.0 / img_cold,
            "final_h_per_run": float(fin["hours"]) / fin["runs"],
            "cold_epochs": e_base, "recipe_epochs": {r: rec["epochs"] for r, rec in (d.get("recipes") or {}).items()},
            "hours": {k: _num((v or {}).get("hours")) for k, v in gh.items()}}, cites


def _chain_hours(rt, recipes, sizes, epochs):
    """sizes: [(cand, null)] per step; the same for every recipe."""
    return sum(rt["seeds"] * (c + n) * epochs[r] * rt["inc_s_per_image_epoch"][r] / 3600.0
               for r in recipes for c, n in sizes)


def estimate_pilot_rebuild(ev, parent, replay_mode="full"):
    rt, cites = rates(ev, parent)
    d = ev.json("%s/exp.json" % parent)
    p0 = d["base"]["n_images"]
    sizes, pool = [], p0
    for s in d["steps"]:
        n = s["n_images"]
        sizes.append((pool + n, pool) if replay_mode == "full" else (2 * n, 2 * n))
        if s.get("clean"):
            pool += n
    recipes = sorted(d["recipes"])
    chains = _chain_hours(rt, recipes, sizes, {r: d["recipes"][r]["epochs"] for r in recipes})
    fixed = sum(rt["hours"].get(k) or 0.0 for k in ("base", "truth", "final"))
    total = chains + fixed
    return round(total, 3), {"estimator": "pilot_rebuild", "from": parent, "replay_mode": replay_mode,
                             "chains_hours": round(chains, 3), "base_truth_final_hours": round(fixed, 3),
                             "cand_null_images": sizes, "upper_bound": replay_mode == "full"}, cites


def _latest_rates(ev):
    """The measured rates of the latest complete pilot; with none in the
    evidence, of the latest finished real loop (its report has the same arms:
    the recovered loop of the funnel audit is priced from the loop whose recipe
    it takes)."""
    from . import diagnose as DG
    for e in reversed(DG.complete_pilots(ev)):
        try:
            return rates(ev, e)
        except Defer:
            continue
    loop, _rep = DG._latest_loop(ev)
    if loop is not None:
        try:
            return rates(ev, loop)
        except Defer:
            pass
    raise Defer("no complete pilot (or finished real loop) in the evidence to measure GPU time per image")


def realloop_sequence(n_verified):
    """(name, clean) of realloop's sequence (realloop.sequence)."""
    v = ["V%d" % i for i in range(1, n_verified + 1)]
    seq = v[:2] + ["UNVERIFIED"] + v[2:3] + ["OTHER_HEAVY"] + v[3:]
    return [(s, s != "UNVERIFIED") for s in seq]


def estimate_realloop(ev, params, menu=None):
    sel = ev.json("step1/select_summary.json")
    n_base = _num(((sel or {}).get("sizes") or {}).get("base_B"))
    if not n_base:
        raise Defer("no Step 1 select build (sizes.base_B) in the evidence to size the loop")
    rt, cites = _latest_rates(ev)
    cites = cites + [ev.cite("step1/select_summary.json", "/sizes/base_B")]
    m = params.get("size") or max(1, int(round(protocol("inc_frac", menu) * n_base)))
    nv = params.get("n_verified") or protocol("n_verified_default", menu)
    recovered = params.get("increment_sources") == SOURCES_RECOVERED
    recipes = params["recipes"].split(",")
    unknown = [r for r in recipes if r not in rt["inc_s_per_image_epoch"]]
    if unknown:
        raise Defer("the pilot measured no GPU time for recipe(s) %s" % unknown)
    seeds = protocol("seeds", menu)
    rt = dict(rt, seeds=seeds)
    ce = rt["cold_epochs"]
    truth = not params.get("no_truth")
    base_h = seeds * n_base * ce * rt["cold_s_per_image_epoch"] / 3600.0
    truth_h, t, sizes, pool = 0.0, n_base, [], n_base
    seq = ([("REC", False)] * protocol("recovered_steps", menu)) if recovered else realloop_sequence(nv)
    for _, clean in seq:
        if truth:
            truth_h += seeds * (t + m) * ce * rt["cold_s_per_image_epoch"] / 3600.0
        sizes.append((pool + m, pool) if params["replay_mode"] == "full" else (2 * m, 2 * m))
        if clean:
            t += m
            pool += m
    chains = _chain_hours(rt, recipes, sizes, rt["recipe_epochs"])
    finals = (len(recipes) + seeds + (seeds if truth else 0)) * rt["final_h_per_run"]
    total = base_h + truth_h + chains + finals
    return round(total, 3), {"estimator": "realloop", "rates_from": rt["from"], "base_images": n_base,
                             "size": m, "n_verified": None if recovered else nv,
                             "steps": len(seq), "recovered": recovered, "truth": truth,
                             "base_hours": round(base_h, 3),
                             "truth_hours": round(truth_h, 3), "chains_hours": round(chains, 3),
                             "final_hours": round(finals, 3), "upper_bound": params["replay_mode"] == "full"}, cites


def estimate_baseline(ev, n_images, menu=None):
    rt, cites = _latest_rates(ev)
    seeds = protocol("seeds", menu)
    h = seeds * n_images * rt["cold_epochs"] * rt["cold_s_per_image_epoch"] / 3600.0 + seeds * rt["final_h_per_run"]
    return round(h, 3), {"estimator": "baseline", "rates_from": rt["from"], "images": n_images}, cites


def estimate_walltime(script):
    """(hours, detail) from the script's #SBATCH --time (one GPU)."""
    p = M.LAB_REPO / script
    try:
        text = E._read_file(p).decode("utf-8")
    except OSError:
        raise Defer("%s not found at %s to read its walltime" % (script, p))
    m = _TIME.search(text)
    if not m:
        raise Defer("%s has no #SBATCH --time" % script)
    days, hh, mm, ss = (int(x) if x else 0 for x in m.groups())
    return round(days * 24 + hh + mm / 60.0 + ss / 3600.0, 3), {"estimator": "walltime", "script": script,
                                                                   "time": m.group(0).split("=", 1)[1].strip()}


def build_job_hours():
    """(hours, detail) of the build job itself, from its own #SBATCH --time.

    Every inc_build_* action runs its builder inside run_inc_build.sh, which
    holds one V100 on GPU-shared for up to that walltime (the allocation has no
    CPU partition: RM-shared fails with "Invalid qos"). The job is charged on
    top of the experiment's runs: at most 4 SU per build at 1.0 SU per V100
    GPU-hour. Read from the script, never assumed (Defer when unreadable)."""
    return estimate_walltime(BUILD_SCRIPT)


def with_build_job(est, info):
    """(est plus the build job's hours, info recording both parts): the price of
    a build is its experiment's runs and the job that builds it."""
    hours, detail = build_job_hours()
    info = dict(info, experiment_hours=est, build_job_hours=hours, build_job=detail)
    return round(float(est) + hours, 3), info


def price(lid, params, ev, parent=None, menu=None):
    """(est_gpu_hours, estimate detail, cites) of lever lid with these params,
    by the estimator levers.json names for it, from GPU time measured in the
    evidence: the price every proposal here carries, and the one a proposal
    from elsewhere (a brain plan) is re-priced with instead of its own figure.
    parent: the pilot an L1 rebuilds. Raises Defer when the evidence cannot
    price it; never returns a guessed or zero price for a build."""
    r = row(lid, menu)
    kind = r.get("estimator")
    if kind == "pilot_rebuild":
        if not parent:
            raise Defer("%s is priced from the pilot it rebuilds; none was given" % lid)
        mode = params.get("replay_mode") or (r.get("fixed") or {}).get("replay_mode") or "full"
        est, info, cites = estimate_pilot_rebuild(ev, parent, mode)
        return with_build_job(est, info) + (cites,)
    if kind == "realloop":
        for k in ("replay_mode", "recipes"):
            if not params.get(k):
                raise Defer("%s is priced from its --%s; none was given" % (lid, k.replace("_", "-")))
        est, info, cites = estimate_realloop(ev, params, menu)
        return with_build_job(est, info) + (cites,)
    if kind == "baseline":
        n = _num(((ev.json("step1/select_summary.json") or {}).get("sizes") or {}).get("base_B"))
        if not n:
            raise Defer("no Step 1 select build (sizes.base_B) in the evidence to price %s" % lid)
        est, info, cites = estimate_baseline(ev, n, menu)
        est, info = with_build_job(est, info)
        return est, info, cites + [ev.cite("step1/select_summary.json", "/sizes/base_B")]
    if kind == "walltime":
        est, info = estimate_walltime(r["script"])
        return est, info, []
    if kind == "zero":
        return 0.0, {"estimator": "zero"}, []
    if kind == "funnel_walltime":
        est, info = estimate_funnel(lid, params, menu)
        return est, info, []
    raise Defer("lever %s names no estimator levers.py knows (%r)" % (lid, kind))


# ------------------------------------------------------------ the funnel audit
# Levers L10-L14 (docs/FUNNEL_AUDIT.md 8.5; runner 5.5.3). L10, L11 and L13 are
# sbatch jobs of run_inc_funnel.sh on the cluster; L11a and L12 run the funnel
# CLI's fetch verb on the lab (R0); L14 is a lab hook that queues verify-only
# tasks for a person. Their paths are derived, never a proposer's: the cluster
# INC_DIR's funnel/prereg_v1.json and funnel/ for the jobs, the lab's INC tree
# for the fetches. A job waits for the file it reads on the cluster (the lever
# row's "preconditions", read from funnel/files.json, the funnel directory's
# listing); until the file is there the proposal carries waits_for and is
# listed under deferred with "then", the way D2 defers L2 behind L3. A
# precondition whose "then" is L11a names the fetch it needs ("what"), and the
# L11a builder fetches what a waiting step needs (_build_fetch).

FUNNEL_LEVERS = ("L10", "L11", "L11a", "L12", "L13", "L14")
# The lab fetches L11a runs, in the order it takes them (inc_funnel_fetch's
# what values but taxonomy, which is L12): the cards whenever their index is
# missing or stale, the others only when a waiting funnel step names them.
FETCH_ORDER = ("cards", "kt7", "known-items", "refetch")
CARDS_INDEX = "funnel/cards/index.json"
# A fetch L11a's command cannot run: funnel fetch refuses --what known-items
# without the H12 source documents (--sources), which neither the L11a command
# nor inc_funnel_fetch's policy row carries.
FETCH_NEEDS = {"known-items": "--sources (the H12 source documents; runner 6.1 F3a: --sources docs/literature)"}
# The order of the funnel's cluster steps and the file each writes (runner 6.1):
# L10's next verb is the first whose output is not on the cluster yet.
FUNNEL_STEPS = (("L10", {"verb": "census"}, "funnel/census_v1.json"),
                ("L10", {"verb": "leak"}, "funnel/leak_v1.json"),
                ("L11", {"part": "geometry"}, "funnel/relation_geometry_v1.json"),
                ("L10", {"verb": "embed-judges"}, "funnel/judges/"),
                ("L10", {"verb": "qualify"}, "funnel/judge_qualification.json"),
                ("L11", {"part": "relation"}, "funnel/relation_audit_v1.json"),
                ("L10", {"verb": "draw"}, "funnel/sample_v1.csv"),
                ("L10", {"verb": "sheets"}, "funnel/sheets_v1/index.json"),
                ("L10", {"verb": "rl-b"}, "funnel/rl_answers/RL-B/"),
                ("L10", {"verb": "ingest"}, "funnel/gold_v1.csv"),
                ("L10", {"verb": "qualify", "rl": 1}, "funnel/rl_qualification.json"),
                ("L10", {"verb": "estimate"}, "funnel/audit_v1.json"))


def cluster_files(ev):
    """{INC_DIR-relative path: {"sha256", "bytes"}} of the funnel directory on
    the cluster (funnel/files.json, remote.py funnel summary), or None when
    the listing is not in the evidence (every file then counts as absent)."""
    lst = ev.json(E.FUNNEL_LISTING)
    files = (lst or {}).get("files") if isinstance(lst, dict) else None
    return files if isinstance(files, dict) else None


def lab_files(ev):
    """The lab's own funnel files the ticker listed (context 'funnel_lab':
    {INC_DIR-relative path: sha256}), or {}."""
    ctx = ev.json(E.CONTEXT) or {}
    v = ctx.get("funnel_lab")
    return v if isinstance(v, dict) else {}


def _has(files, path):
    """True when `path` (a file, or a directory ending in '/') is in the listing."""
    if not files:
        return False
    if path.endswith("/"):
        return any(str(k).startswith(path) for k in files)
    return path in files


# The copy detector's record (contract docs/FUNNEL_AUDIT.md 14 A2): once the
# pre-registration requires copy detector version v > 1 (the ticker's context
# funnel_leak, read from the lab's copy of the prereg by campaign), the leak
# step writes funnel/leak_v<v>.json, and its lineage key carries detector=v,
# so the version 1 run the lineage holds does not stand for it (as L11a's
# config key does for a stale cards index).
LEAK_OUTPUT = "funnel/leak_v%d.json"


def leak_detector(ev):
    """(the copy detector version the funnel's leak step must produce, the
    context record funnel_leak or None): the context's detector_version when
    it is an integer >= 1, else 1."""
    rec = (ev.json(E.CONTEXT) or {}).get("funnel_leak") if ev is not None else None
    rec = rec if isinstance(rec, dict) else None
    v = (rec or {}).get("detector_version")
    return (v if isinstance(v, int) and not isinstance(v, bool) and v >= 1 else 1), rec


def funnel_steps(ev):
    """FUNNEL_STEPS with the leak step of the detector version the prereg
    requires (leak_detector): its output funnel/leak_v<v>.json and, above
    version 1, its lineage key detector=v."""
    v, _rec = leak_detector(ev)
    out = []
    for lv, params, path in FUNNEL_STEPS:
        if lv == "L10" and params.get("verb") == "leak" and v > 1:
            params, path = dict(params, detector=v), LEAK_OUTPUT % v
        out.append((lv, params, path))
    return tuple(out)


def funnel_next(ev, lid=None):
    """(lever, params, output) of the funnel's first cluster step whose output
    is not in the cluster listing, among steps of lever `lid` (default: any);
    (None, None, None) when every step's output is there. The leak step is
    the one of the detector version the prereg requires (funnel_steps)."""
    files = cluster_files(ev)
    for lv, params, out in funnel_steps(ev):
        if lid is not None and lv != lid:
            continue
        if not _has(files, out):
            return lv, dict(params), out
    return None, None, None


def _funnel_paths(ev, menu, lab=False):
    """{"prereg", "out", "inc"}: the funnel's paths on the cluster (INC_DIR from
    the evidence, levers.inc_dir) or on the lab (model.LAB_REPO)."""
    inc = str(M.LAB_REPO / protocol("funnel_lab_inc_dir", menu)) if lab else inc_dir(ev)
    return {"inc": inc, "prereg": posixpath.join(inc, protocol("funnel_prereg", menu)),
            "out": posixpath.join(inc, protocol("funnel_out", menu))}


def funnel_resources(verb, menu=None):
    """The extra sbatch flags of a funnel verb (protocol funnel_sbatch_resources
    of funnel_verb_class[verb]), or None when there are none."""
    cls = protocol("funnel_verb_class", menu).get(verb)
    if cls is None:
        raise LeverError("%r is not a funnel verb" % (verb,))
    flags = protocol("funnel_sbatch_resources", menu).get(cls)
    if flags is None:
        raise LeverError("funnel verb class %r has no sbatch resources entry" % cls)
    return list(flags) or None


def estimate_funnel(lid, params, menu=None):
    """(hours, detail): a funnel job holds run_inc_funnel.sh's GPU for up to its
    #SBATCH --time. A verb of the gpu_large class (rl-b) holds an H100 instead
    of the header's V100: its hours are stated in V100 GPU-hours at the
    su_rates.json ratio, since the policy row prices est_gpu_hours on V100."""
    r = row(lid, menu)
    hours, info = estimate_walltime(r.get("script") or protocol("funnel_script", menu))
    verb = params.get("verb") if lid == "L10" else {"L11": "map", "L13": "recover"}.get(lid)
    cls = protocol("funnel_verb_class", menu).get(verb)
    gpu = protocol("funnel_gpu_by_class", menu).get(cls)
    info = dict(info, estimator="funnel_walltime", verb=verb, verb_class=cls, gpu=gpu)
    if gpu and gpu != "v100":
        from ..brain import su_ledger
        rates = su_ledger.rates()
        ratio = float(rates[gpu]["su_per_gpu_hour"]) / float(rates["v100"]["su_per_gpu_hour"])
        info.update(gpu_hours=hours, su_ratio_to_v100=ratio)
        hours = round(hours * ratio, 3)
    return hours, info


def _funnel_once(ev, lid, key, menu=None):
    """Defer when the campaign's lineage records lever lid with these params
    (a funnel step runs once per campaign; a person decides whether to repeat
    one that ran and left no output). A run whose Slurm job ended in a failure
    state (the ticker's lineage status 'failed' with its 'job' record:
    campaign.JOB_FAILED_STATES, read from sacct) does not count: the step is
    proposed again, at most protocol funnel_job_retries times; the next failed
    run leaves it with a person, with the last job's state and log."""
    ctx = ev.json(E.CONTEXT) or {}
    failed = []
    for i, r in enumerate(ctx.get("lineage") or []):
        if not isinstance(r, dict) or r.get("lever") != lid:
            continue
        p = r.get("params") if isinstance(r.get("params"), dict) else {}
        if not all(p.get(k) == v for k, v in key.items()):
            continue
        if r.get("status") == "failed" and isinstance(r.get("job"), dict):
            failed.append(r)
            continue
        if r.get("status") in LINEAGE_IGNORED:
            continue
        raise Defer("%s %s was already run in this campaign (%s); its output is not on the cluster yet, and a "
                    "person decides whether to run it again" % (lid, " ".join("%s=%s" % kv for kv in sorted(
                        key.items())), r.get("status")))
    retries = int(protocol("funnel_job_retries", menu))
    if len(failed) > retries:
        job = failed[-1]["job"]
        states = job.get("states") if isinstance(job.get("states"), dict) else {}
        raise Defer("%s %s failed %d times in this campaign (the last: job %s ended %s); a step is proposed again "
                    "at most %d times after a failed job, so a person reads the last job's log (%s) and decides "
                    "whether to run it again" % (lid, " ".join("%s=%s" % kv for kv in sorted(key.items())),
                                                 len(failed), ",".join(str(j) for j in job.get("ids") or []) or "?",
                                                 ",".join(str(states[j]) for j in sorted(states)) or "in a failure state",
                                                 retries, job.get("log") or "not recorded"))


def _preconditions(lid, key, menu):
    """The row's preconditions of a funnel step (keyed by verb or part), as a
    list: a single precondition or several, in the order they are checked."""
    pre = (row(lid, menu).get("preconditions") or {}).get(key)
    if isinstance(pre, dict):
        return [pre]
    return [x for x in pre if isinstance(x, dict)] if isinstance(pre, list) else []


def _waits(ev, lid, key, menu):
    """The waits_for record of a funnel job whose input file is not on the
    cluster (the first of the row's preconditions keyed by verb or part whose
    file the listing lacks), or None. A precondition met by a lab fetch names
    its kind ('what'), which the record carries."""
    for pre in _preconditions(lid, key, menu):
        if _has(cluster_files(ev), pre["cluster_file"]):
            continue
        w = {"lever": pre.get("then"), "cluster_file": pre["cluster_file"], "why": pre.get("why"),
             "listing": "in the evidence" if cluster_files(ev) is not None else "not in the evidence"}
        if pre.get("what"):
            w["what"] = pre["what"]
        return w
    return None


def fetch_waits(ev, menu=None):
    """{what: waits record}: the lab fetches (L11a) the funnel's next cluster
    steps wait for. For the next step of L10 and of L11 (funnel_next, whether
    it is proposed now or deferred because it already ran), every precondition
    whose file the cluster lacks and whose 'then' is L11a, keyed by its 'what',
    with the step that waits ('for')."""
    menu = menu or load_menu()
    out = {}
    for lid in ("L10", "L11"):
        nl, params, _out = funnel_next(ev, lid)
        if nl is None:
            continue
        key = params["verb"] if lid == "L10" else params["part"]
        for pre in _preconditions(lid, key, menu):
            if pre.get("then") != "L11a" or not pre.get("what") or _has(cluster_files(ev), pre["cluster_file"]):
                continue
            out.setdefault(pre["what"], {"lever": "L11a", "what": pre["what"], "cluster_file": pre["cluster_file"],
                                         "why": pre.get("why"), "for": dict(params, lever=lid)})
    return out


def _funnel_proposal(lid, params, derived, d, cites, menu, waits=None, extra=None):
    est, info = (estimate_funnel(lid, params, menu) if row(lid, menu).get("estimator") == "funnel_walltime"
                 else (0.0, {"estimator": "zero"}))
    p = _proposal(lid, params, derived, d, cites, est, info, None, None, menu)
    if waits:
        p["waits_for"] = waits
    if extra:
        p.update(extra)
    return p


def recovered_loop_params(ev, menu=None):
    """(params, parent, cites) of the recovered real loop (realloop_v2,
    contract 9.1): the latest finished real loop's base, replay mode, recipes
    and gate flips mode, --increment-sources recovered, --step1-overlay
    <INC_DIR>/step1_r1 and --size its increment_images. Defers until the
    recovery overlay is complete (funnel/recovery.json status complete)."""
    from . import diagnose as DG
    rec = ev.json(E.FUNNEL_RECOVERY)
    if not isinstance(rec, dict) or rec.get("status") != "complete":
        raise Defer("the recovery overlay is not complete (%s status %r)"
                    % (E.FUNNEL_RECOVERY, (rec or {}).get("status") if isinstance(rec, dict) else None))
    loop, _rep = DG._latest_loop(ev)
    if loop is None:
        raise Defer("no finished real loop in the evidence to take the recipe from")
    pd = ev.json("%s/exp.json" % loop) or {}
    art = "%s/exp.json" % loop
    size = pd.get("increment_images")
    if not isinstance(size, int) or isinstance(size, bool):
        raise Defer("%s records no increment_images" % loop)
    src = (pd.get("base") or {}).get("source_manifest")
    base = src if isinstance(src, str) and src.startswith("/") else posixpath.join(inc_dir(ev, loop), "step1",
                                                                                   "base_B.jsonl")
    params = {"base": base, "replay_mode": pd.get("replay_mode", DG.DEFAULT_REPLAY_MODE),
              "recipes": _recipes_param(pd.get("recipes") or {}), "increment_sources": SOURCES_RECOVERED,
              "step1_overlay": posixpath.join(inc_dir(ev, loop), protocol("funnel_recover_out", menu).rstrip("/")),
              "size": size}
    gp, gc = gate_params(ev, loop, menu)
    params.update(gp)
    cites = [ev.cite(E.FUNNEL_RECOVERY, "/status"), ev.cite(art, "/increment_images")] + list(gc)
    for ptr in ("/base/source_manifest", "/replay_mode", "/recipes"):
        try:
            cites.append(ev.cite(art, ptr))
        except KeyError:
            pass
    _once(ev, "L2", loop, params=params)
    params["exp"] = realloop_name(ev.exps())
    return params, loop, cites


def _build_funnel(lid, d, ev, menu):
    """[proposal] of funnel lever lid triggered by diagnosis d."""
    det = d.get("detail") or {}
    cites = list(d["cites"])
    if lid == "L10":
        nl, params, out = funnel_next(ev, "L10")
        if nl is None:
            raise Defer("every L10 stage's output is on the cluster (funnel/files.json)")
        if det.get("first_verb") and cluster_files(ev) is None:
            params = {"verb": det["first_verb"]}
        _funnel_once(ev, "L10", params)
        paths = _funnel_paths(ev, menu)
        derived = {"prereg": paths["prereg"], "out": paths["out"],
                   "sbatch_resources": funnel_resources(params["verb"], menu)}
        step = {"writes": out}
        if params.get("detector"):
            # the prereg's amendment supersedes the version 1 record: the leak step runs again
            _v, rec = leak_detector(ev)
            step.update(detector_version=params["detector"], supersedes=LEAK_OUTPUT % 1,
                        amendment=(rec or {}).get("amendment"))
            try:
                cites.append(ev.cite(E.CONTEXT, "/funnel_leak/detector_version"))
            except KeyError:
                pass
        return [_funnel_proposal("L10", params, derived, d, cites, menu,
                                 waits=_waits(ev, "L10", params["verb"], menu),
                                 extra={"funnel_step": step})]
    if lid == "L11":
        nl, params, out = funnel_next(ev, "L11")
        if nl is None:
            raise Defer("both L11 parts' outputs are on the cluster (funnel/files.json)")
        _funnel_once(ev, "L11", params)
        paths = _funnel_paths(ev, menu)
        return [_funnel_proposal("L11", params, {"prereg": paths["prereg"], "out": paths["out"]}, d, cites, menu,
                                 waits=_waits(ev, "L11", params["part"], menu),
                                 extra={"funnel_step": {"writes": out}, "sources": det.get("s2_sources") or []})]
    if lid == "L11a":
        return [_build_fetch(d, ev, menu)]
    if lid == "L12":
        target = "funnel/taxonomy_cache.json"
        key = {"what": "taxonomy"}
        if _has(cluster_files(ev), target):
            raise Defer("%s is already on the cluster (funnel/files.json)" % target)
        if target in lab_files(ev):
            raise Defer("%s is on the lab and not yet on the cluster: the funnel sync (inc_funnel_sync) pushes it"
                        % target)
        _funnel_once(ev, lid, key)
        paths = _funnel_paths(ev, menu, lab=True)
        derived = {"prereg": paths["prereg"], "out": paths["out"],
                   "names_from": posixpath.join(paths["inc"], "step1", "pool_summary.json")}
        return [_funnel_proposal(lid, key, derived, d, cites, menu,
                                 extra={"sources": det.get("s2_sources") or [], "writes": target, "stale": None})]
    if lid == "L13":
        policy = det.get("policy")
        if not policy:
            raise Defer("D18 names no recovery policy")
        _funnel_once(ev, "L13", {"policy": policy})
        paths = _funnel_paths(ev, menu)
        inc = paths["inc"]
        derived = {"prereg": paths["prereg"], "audit": posixpath.join(inc, "funnel", "audit_v1.json"),
                   "maps": posixpath.join(inc, "funnel", "class_maps.json"),
                   "out": posixpath.join(inc, protocol("funnel_recover_out", menu))}
        return [_funnel_proposal("L13", {"policy": policy}, derived, d, cites, menu,
                                 extra={"strata": det.get("strata") or [],
                                        "known_confusions": det.get("known_confusions") or []})]
    if lid == "L14":
        rows = det.get("to_l14") or []
        if not rows:
            raise Defer("no class map waits for a person's verification")
        _funnel_once(ev, "L14", {})
        return [_funnel_proposal("L14", {}, {}, d, cites, menu, extra={"queue": rows})]
    raise Defer("no builder for funnel lever %s" % lid)


def _build_fetch(d, ev, menu):
    """The L11a proposal: the first lab fetch due, in FETCH_ORDER. The cards
    when their index is stale (context funnel_cards_stale: fetched again once
    per domain config, the config's sha12 in the lineage key) or on neither
    host; then a fetch a waiting funnel step names (fetch_waits: kt7 while
    embed-judges, qualify, draw, sheets or estimate waits for the KT7 photos;
    known-items while a step waits for the H12 list). A fetch whose file is on
    the lab and not yet on the cluster waits for the sync, one that already
    ran waits for a person (_funnel_once), and the next fetch due is taken
    instead; with none left the lever is deferred with every reason."""
    det = d.get("detail") or {}
    cites = list(d["cites"])
    stale = (ev.json(E.CONTEXT) or {}).get("funnel_cards_stale")
    waits = fetch_waits(ev, menu)
    held = []
    for what in FETCH_ORDER:
        key = {"what": what}
        target = CARDS_INDEX if what == "cards" else (waits.get(what) or {}).get("cluster_file")
        if what == "cards" and isinstance(stale, dict) and stale.get("domain_sha256"):
            # the cards index recorded refusals under another domain config: fetch
            # again, once per config (the lineage key carries the config's sha)
            key["config"] = str(stale["domain_sha256"])[:12]
        elif target is None:
            continue                              # no waiting funnel step names this fetch
        elif _has(cluster_files(ev), target):
            held.append("%s is already on the cluster (funnel/files.json)" % target)
            continue
        elif target in lab_files(ev):
            held.append("%s is on the lab and not yet on the cluster: the funnel sync (inc_funnel_sync) pushes it"
                        % target)
            continue
        try:
            _funnel_once(ev, "L11a", key, menu)
        except Defer as e:
            held.append(str(e))
            continue
        if what in FETCH_NEEDS:
            w = waits.get(what) or {}
            held.append("%s waits for %s, which fetch --what %s writes only with %s; the L11a command carries no "
                        "such flag, so a person runs that fetch on the lab" % (
                            "L10 %s" % (w.get("for") or {}).get("verb") if (w.get("for") or {}).get("verb")
                            else "a funnel step", target, what, FETCH_NEEDS[what]))
            continue
        paths = _funnel_paths(ev, menu, lab=True)
        derived = {"prereg": paths["prereg"], "out": paths["out"]}
        extra = {"sources": (det.get("s2_sources") or []) if what == "cards" else [], "writes": target,
                 "stale": stale if key.get("config") else None}
        if what != "cards" and what in waits:
            extra["needed_by"] = waits[what]["for"]      # the waiting step this fetch unblocks
        return _funnel_proposal("L11a", key, derived, d, cites, menu, extra=extra)
    raise Defer("; ".join(held) or "no lab fetch is due (%s)" % ", ".join(FETCH_ORDER))


def verify_queue_rows(ev):
    """The verify tasks of lever L14: class maps whose card and source class
    counts disagree (class_maps.json proposals with status to_L14)."""
    maps = ev.json(E.FUNNEL_CLASS_MAPS) if ev is not None else None
    rows = []
    for p in (maps or {}).get("proposals") or [] if isinstance(maps, dict) else []:
        if isinstance(p, dict) and p.get("status") == "to_L14":
            rows.append({k: p.get(k) for k in ("source", "src_id", "src_name", "map_to", "via", "reason")})
    return rows


def funnel_sync_needed(ev):
    """The funnel files one side holds and the other does not (runner 6.3),
    [INC_DIR-relative path]: the lab's push-list files the cluster lacks
    (context funnel_lab against funnel/files.json), and the cluster's
    pull-list files the lab lacks or holds another version of (funnel/files.json
    and the shipped recovery record against context funnel_lab_pull). A
    cluster file listed without a sha256 (too large to hash there) counts
    only while the lab has no copy. A machine-local crop table
    (funnel.fetch.MACHINE_LOCAL: kt7/crops_kt7.csv, refetch/crops_refetch.csv)
    counts only while the cluster has none: it holds each host's own image
    paths, and fetch.check_manifest rebuilds it on arrival, so the two copies
    never hash alike."""
    from . import executor as X
    from ..funnel import fetch as FF
    have = cluster_files(ev) or {}
    machine = {"funnel/" + rel for rel in FF.MACHINE_LOCAL}
    out = {p for p, sha in lab_files(ev).items()
           if (p not in have if p in machine else (have.get(p) or {}).get("sha256") != sha)}
    ctx = ev.json(E.CONTEXT) or {}
    local = ctx.get("funnel_lab_pull") if isinstance(ctx.get("funnel_lab_pull"), dict) else {}
    for p, info in have.items():
        if not X.funnel_pull_refusals([p]):
            sha = (info or {}).get("sha256") if isinstance(info, dict) else None
            if p not in local or (sha is not None and local[p] != sha):
                out.add(p)
    rec = ev.provenance.get(E.FUNNEL_RECOVERY) or {}
    rel = X.FUNNEL_PULL_RECOVERY
    if ev.json(E.FUNNEL_RECOVERY) is not None and rec.get("sha256") and local.get(rel) != rec.get("sha256"):
        out.add(rel)
    return sorted(out)


# --------------------------------------------------------------- propose
def _lit(r):
    return [{"paper_id": x.get("paper_id"), "line": None, "quote": None, "doc_ref": x.get("doc_ref"),
             "status": x.get("status", "corpus entry pending")} for x in r.get("lit") or []]


def _proposal(lid, params, derived, trigger, cites, est, est_detail, parent, child, menu):
    r = row(lid, menu)
    params = dict(params, **{k: v for k, v in (r.get("fixed") or {}).items()})
    params["est_gpu_hours"] = est
    if r.get("argv") is None and r.get("kind") == "lab_hook":
        # A lab hook (L14) has no command: the executor calls the hook the
        # campaign registered for its policy action, with these params.
        ok, reasons = check_params(lid, params, menu)
        if not ok:
            raise LeverError("%s params refused: %s" % (lid, "; ".join(reasons)))
        cmd = []
    else:
        cmd = argv(lid, params, derived, menu)
    p = M.proposal(lid, cmd, params, r["policy_action"], r["risk"], [trigger["id"]], list(cites),
                   lit=_lit(r), control=r.get("control", ""), success=r.get("success", ""),
                   falsifier=r.get("falsifier", ""), est_gpu_hours=est)
    p.update(predicted=r.get("predicted"), parent_exp=parent, child_exp=child, via=r.get("via"),
             estimate=est_detail, title=r.get("title"))
    return p


# ---------------------------------------------------------------- lineage
def _defn(ev, e):
    return ev.json("%s/exp.json" % e) or {}


def _recipe_names(recipes):
    return sorted(recipes or ())


def _after(ev, parent):
    """Experiments initialised after parent (Evidence.exps order)."""
    exps = ev.exps()
    return exps[exps.index(parent) + 1:] if parent in exps else []


def _loops(ev, exps=None):
    """Real loops (exp.json builder inc.realloop build) among exps (default: all)."""
    from . import diagnose as DG
    return [e for e in (ev.exps() if exps is None else exps) if _defn(ev, e).get("builder") == DG.REALLOOP_BUILDER]


def _exp_cite(ev, e, *ptrs):
    """A cite of e's exp.json at the first pointer it holds."""
    for ptr in ptrs:
        try:
            return ev.cite("%s/exp.json" % e, ptr)
        except KeyError:
            continue
    return None


def _recorded(ev, lids, parent):
    """(child, cite) of the first context lineage record of a lever in lids
    applied to parent (evidence.py 'lineage'), else (None, None)."""
    ctx = ev.json(E.CONTEXT) or {}
    for i, r in enumerate(ctx.get("lineage") or []):
        if not isinstance(r, dict) or r.get("status") in LINEAGE_IGNORED:
            continue
        if r.get("lever") in lids and r.get("parent_exp") == parent:
            return r.get("child_exp") or "an in-flight build", ev.cite(E.CONTEXT, "/lineage/%d" % i)
    return None, None


def _same_base(ev, d):
    """False only when both the loop's base sha256 and the current Step 1
    base_B.jsonl sha256 are known and differ (a new select build)."""
    want = (((ev.json("step1/select_summary.json") or {}).get("outputs") or {}).get("base_B.jsonl") or {}).get("sha256")
    got = (d.get("base") or {}).get("manifest_sha256")
    return not (want and got) or want == got


def _on_gate(ev, e, want):
    """True when experiment e is decided on flips mode `want` (None: any)."""
    return want is None or gate_mode(ev, e) == want


def applied(ev, lid, parent, detail=None, params=None):
    """(experiment, cite) showing lever lid was already applied to parent,
    else (None, None). detail: D4's detail for an L2/L6 from a pilot's
    decision (its replay_mode and recipes); params: the loop flags of an L2
    or L5 rebuild of a real loop. An earlier experiment counts only when it
    is decided on the gate the lever forwards (gate_mode): the parent's for
    L1, L2 from D1 and L5, the pilot D4 evaluated for L2 and L6 from D4 (the
    parent); L9's own, net. See the module doc, 'Lineage'."""
    from . import diagnose as DG
    lids = ("L2", "L6") if lid in ("L2", "L6") else (lid,)
    child, c = _recorded(ev, lids, parent)
    if c is not None:
        return child, c
    if lid in ("L1", "L2", "L5", "L6") and parent is not None:
        want_gate = gate_mode(ev, parent)
    else:
        want_gate = None
    if lid == "L1":
        for e in _after(ev, parent):
            if e in DG.pilots(ev) and _on_gate(ev, e, want_gate):
                mode, mc = DG._replay_mode(ev, e)
                if mode == "full":
                    return e, mc
        return None, None
    if lid in ("L2", "L6") and detail is not None:
        want = (detail.get("replay_mode"), _recipe_names(detail.get("recipes")))
        for e in _loops(ev):
            d = _defn(ev, e)
            if (d.get("replay_mode", DG.DEFAULT_REPLAY_MODE), _recipe_names(d.get("recipes"))) == want \
                    and _same_base(ev, d) and _on_gate(ev, e, want_gate):
                return e, _exp_cite(ev, e, "/replay_mode", "/builder")
        return None, None
    if lid in ("L2", "L5") and params is not None:
        pd = _defn(ev, parent)
        for e in _loops(ev, _after(ev, parent)):
            d = _defn(ev, e)
            if _recipe_names(d.get("recipes")) != _recipe_names(pd.get("recipes")) or not _on_gate(ev, e, want_gate):
                continue
            mode = d.get("replay_mode", DG.DEFAULT_REPLAY_MODE)
            if lid == "L2" and mode == params.get("replay_mode"):
                return e, _exp_cite(ev, e, "/replay_mode", "/builder")
            size, cur = d.get("increment_images"), pd.get("increment_images")
            if lid == "L5" and mode == params.get("replay_mode") and isinstance(size, int) \
                    and isinstance(cur, int) and size > cur:
                return e, _exp_cite(ev, e, "/increment_images")
        return None, None
    if lid == "L8" and BASELINE_B_EXP in ev.exps():
        return BASELINE_B_EXP, _exp_cite(ev, BASELINE_B_EXP, "/exp", "/type")
    if lid == "L9":
        want = (row("L9").get("fixed") or {}).get("gate_flips_mode")
        for e in _after(ev, parent):
            if e in DG.pilots(ev):
                mode, mc, _src = DG.pinned_gate(ev, e)
                if mode == want:
                    return e, mc or _exp_cite(ev, e, "/gate/flips_mode", "/builder")
        return None, None
    return None, None


def _once(ev, lid, parent, detail=None, params=None):
    """Defer when applied() finds lever lid already applied to parent."""
    child, _c = applied(ev, lid, parent, detail, params)
    if child is not None:
        raise Defer("%s was already applied to %s: %s (a lever is applied to a parent once; a person decides "
                    "whether to repeat it)" % (lid, parent or "the campaign", child))


# ------------------------------------------------------------ build flags
def loop_params(ev, parent):
    """(params, cites): the realloop build flags of the real loop `parent`
    as exp.json and build_summary.json record them, for a rebuild of the same
    loop: --base its source manifest (INC_DIR/step1/base_B.jsonl, not the
    experiment's own copy, which has no select_summary.json beside it),
    --replay-mode, --recipes, its relevance criterion (--relevance, its
    step1.relevance.path; or --increment-sources evidence when exp.json
    step1.increment_sources.mode says so), --size (increment_images),
    --n-verified (build_summary n_verified, else the verified steps),
    no_truth when it was built without the truth arm, and --gate-flips-mode
    when its pinned flips mode is not the builders' default (gate_params).
    An evidence loop built with a --min-evidence other than the protocol's
    default is not rebuilt (Defer: the flag is not on the menu)."""
    from . import diagnose as DG
    pd = ev.json("%s/exp.json" % parent)
    if pd is None:
        raise Defer("no %s/exp.json in the evidence" % parent)
    if pd.get("builder") != DG.REALLOOP_BUILDER:
        raise Defer("%s is not a real loop (builder %r)" % (parent, pd.get("builder")))
    art = "%s/exp.json" % parent
    cites = []
    src = (pd.get("base") or {}).get("source_manifest")
    if isinstance(src, str) and src.startswith("/") and src.endswith(".jsonl"):
        base = src
        cites.append(ev.cite(art, "/base/source_manifest"))
    else:
        base = posixpath.join(inc_dir(ev, parent), "step1", "base_B.jsonl")
        cites.append(ev.cite(art, "/base/manifest"))
    params = {"base": base, "replay_mode": pd.get("replay_mode", DG.DEFAULT_REPLAY_MODE),
              "recipes": _recipes_param(pd.get("recipes") or {})}
    if "replay_mode" in pd:
        cites.append(ev.cite(art, "/replay_mode"))
    crit = (pd.get("step1") or {}).get("increment_sources")
    mode = crit.get("mode") if isinstance(crit, dict) else None
    if mode == SOURCES_RECOVERED:
        raise Defer("%s draws its increments from the funnel's recovery overlay; only the funnel's own L2 builds "
                    "such a loop (recovered_loop_params)" % parent)
    if crit is not None and mode not in protocol("increment_sources_modes"):
        raise Defer("%s records increment sources %r, which realloop build --increment-sources does not take"
                    % (parent, crit))
    if mode == SOURCES_EVIDENCE:
        if crit.get("min_evidence") != protocol("min_evidence_default"):
            raise Defer("%s was built with --min-evidence %r; the menu rebuilds only the protocol's default (%r)"
                        % (parent, crit.get("min_evidence"), protocol("min_evidence_default")))
        params["increment_sources"] = SOURCES_EVIDENCE
        cites.append(ev.cite(art, "/step1/increment_sources/mode"))
    rel = ((pd.get("step1") or {}).get("relevance") or {}).get("path")
    if rel and mode != SOURCES_EVIDENCE:
        params["relevance"] = rel
        cites.append(ev.cite(art, "/step1/relevance/path"))
    size = pd.get("increment_images")
    if isinstance(size, int) and not isinstance(size, bool):
        params["size"] = size
        cites.append(ev.cite(art, "/increment_images"))
    bs = ev.json("%s/build_summary.json" % parent) or {}
    nv = bs.get("n_verified")
    if isinstance(nv, int) and not isinstance(nv, bool):
        params["n_verified"] = nv
        cites.append(ev.cite("%s/build_summary.json" % parent, "/n_verified"))
    else:
        ver = [i for i, st in enumerate(pd.get("steps") or []) if st.get("kind") == KIND_VERIFIED]
        if ver:
            params["n_verified"] = len(ver)
            cites += [ev.cite(art, "/steps/%d/kind" % i) for i in ver]
    if pd.get("truth") is False:
        params["no_truth"] = 1
        cites.append(ev.cite(art, "/truth"))
    gp, gc = gate_params(ev, parent)
    params.update(gp)
    cites += gc
    return params, cites


def _l2_params(ev, d, menu, size=None, n_verified=None):
    """(params, pilot, cites) of the real loop D4's decision on its pilot
    builds: the replay mode and recipes, Step 1's base, the relevance
    criterion (increment_criterion: --relevance with a relevance.json made
    for this select build that passed its calibration check, or
    --increment-sources evidence when that file failed its own check and
    the evidenced pool holds the build's increments at this size and
    n_verified; with both None, at the defaults or, when those do not fit,
    at N and M sized by the R4 rule, which then join the params as --size
    and --n-verified where they differ from the defaults), and the pilot's
    pinned flips mode as --gate-flips-mode when it is not the default.
    cites: the criterion's
    (the relevance file's failed check and the evidence it rests on; none
    for --relevance). Defers while D4's choice is withheld (D1, D15 or D16
    blocks it on the pilot), when a loop of that decision exists (applied),
    or when no criterion can build it (increment_criterion)."""
    det = d.get("detail") or {}
    pilot = det.get("pilot") or d.get("exp")
    if det.get("blocked_by"):
        b = det["blocked_by"]
        if b.get("pending"):
            raise Defer("D4's recipe choice on %s is withheld: %s (%s; L2 waits for the final check)"
                        % (pilot, b.get("summary"), b.get("id")))
        raise Defer("D4's recipe choice on %s is withheld: %s fired on it (%s first)"
                    % (pilot, b.get("id"), " + ".join(d.get("levers") or []) or "a person"))
    _once(ev, "L2", pilot, detail=det)
    inc = inc_dir(ev, pilot)
    crit, _rec, cites = increment_criterion(ev, inc, n_verified=n_verified, size=size, menu=menu)
    params = {"exp": realloop_name(ev.exps()), "base": posixpath.join(inc, "step1", "base_B.jsonl"),
              "replay_mode": det["replay_mode"], "recipes": _recipes_param(det["recipes"])}
    params.update(crit)                                 # --relevance P, or --increment-sources evidence
    #                                                     (and a sized loop's --size / --n-verified)
    params.update(gate_params(ev, pilot, menu)[0])      # the gate the pilot D4 evaluated was decided with
    return params, pilot, cites


def _rebuild_params(ev, parent, lid, **change):
    """(params, cites) of a rebuild of the real loop `parent` with `change`
    (L2 from D1: replay_mode full; L5: a larger size), applied once."""
    params, cites = loop_params(ev, parent)
    if lid == "L2" and params.get("no_truth"):
        raise Defer("%s has no truth arm, so D1 cannot have read one; its rebuild is not L2" % parent)
    params.update(change)
    _once(ev, lid, parent, params=params)
    cites = cites + check_criterion(ev, params, inc_dir(ev, parent))
    params["exp"] = child_name(parent, ev.exps())
    return params, cites


def _build(lid, d, ev, menu, fired):
    """[proposal] for lever lid triggered by diagnosis d."""
    det = d.get("detail") or {}
    if lid in FUNNEL_LEVERS:
        return _build_funnel(lid, d, ev, menu)
    if lid == "L1":
        parent = det.get("pilot") or d.get("exp")
        _once(ev, "L1", parent)
        child = child_name(parent, ev.exps())
        gp, gc = gate_params(ev, parent, menu)
        params = dict({"exp": child}, **gp)
        est, info, c = price(lid, params, ev, parent, menu)
        return [_proposal(lid, params, {}, d, d["cites"] + gc + c, est, info, parent, child, menu)]
    if lid == "L9":
        return [_build_l9(d, ev, menu)]
    if lid == "L2" and d["id"] == "D18":
        params, parent, pc = recovered_loop_params(ev, menu)
        est, info, c = price(lid, params, ev, parent, menu)
        return [_proposal(lid, params, {}, d, d["cites"] + pc + c, est, info, parent, params["exp"], menu)]
    if lid in ("L2", "L6"):
        if d["id"] == "D11":
            d4 = fired.get("D4")
            if not d4 or d4["name"] != "decision_slot_ready":
                raise Defer("L6 follows a ready D4: no recipe choice to build without the truth arm")
            src = d4
        else:
            src = d
        cites = list(d["cites"])
        if src["id"] == "D1":                       # a real loop in sample mode: the same loop, full replay
            if lid != "L2":
                raise Defer("D1 on a real loop rebuilds it with L2 only")
            parent = d.get("exp")
            params, pc = _rebuild_params(ev, parent, "L2", replay_mode="full")
            cites += pc
        else:
            params, parent, cc = _l2_params(ev, src, menu)
            cites += gate_params(ev, parent, menu)[1]
            have = {_key(x) for x in cites}
            cites += [x for x in cc if _key(x) not in have]      # the relevance criterion's evidence
        if lid == "L6":
            params["no_truth"] = 1
        est, info, c = price(lid, params, ev, parent, menu)
        return [_proposal(lid, params, {}, d, cites + c, est, info, parent, params["exp"], menu)]
    if lid == "L3":
        _l3_ok(ev)
        _once(ev, "L3", None)
        est, info, c = price(lid, {}, ev, None, menu)
        return [_proposal(lid, {}, {}, d, d["cites"], est, info, None, None, menu)]
    if lid == "L4":
        exp = d.get("exp") or ev.exp
        _once(ev, "L4", exp)
        derived, c = audit_args(ev, exp)
        est, info, _ = price(lid, {"exp": exp}, ev, exp, menu)
        return [_proposal(lid, {"exp": exp}, derived, d, d["cites"] + c, est, info, exp, None, menu)]
    if lid == "L5":
        parent = d.get("exp")
        if not det.get("size"):
            raise Defer("L5 rebuilds a real loop with a larger --size; D8 gave no size for %s" % parent)
        cur = _defn(ev, parent).get("increment_images")
        if isinstance(cur, int) and int(det["size"]) <= cur:
            raise Defer("L5 is a larger --size: %d is not above %s's %d" % (int(det["size"]), parent, cur))
        params, pc = _rebuild_params(ev, parent, "L5", size=int(det["size"]))
        est, info, c = price(lid, params, ev, parent, menu)
        return [_proposal(lid, params, {}, d, d["cites"] + pc + c, est, info, parent, params["exp"], menu)]
    if lid == "L7":
        out = []
        for u in det.get("units") or []:
            if u.get("transient"):
                out.append(_proposal(lid, {"exp": d.get("exp") or ev.exp, "unit": u["unit"], "cause": u["cause"]},
                                     {}, d, d["cites"], 0.0, {"estimator": "zero"}, d.get("exp"), None, menu))
        return out
    if lid == "L8":
        _once(ev, "L8", None)
        params = {"exp": BASELINE_B_EXP, "manifest": posixpath.join(inc_dir(ev), "step1", "base_B.jsonl")}
        est, info, c = price(lid, params, ev, None, menu)
        return [_proposal(lid, params, {}, d, d["cites"] + c, est, info, None, BASELINE_B_EXP, menu)]
    raise Defer("no builder for lever %s" % lid)


def _build_l9(d, ev, menu):
    """L9 from D15: the parent pilot rebuilt with its own replay mode and the
    gate L9 fixes (net). Defers on a parent that is not a pilot, one already
    pinned to that gate (D15 gives card X6 there), or one L9 was already
    applied to (a later pilot pinned to net, or a lineage record)."""
    from . import diagnose as DG
    parent = d.get("exp")
    pd = _defn(ev, parent)
    if pd.get("builder") != DG.PILOT_BUILDER:
        raise Defer("L9 rebuilds a pilot; %s is not one (builder %r)" % (parent, pd.get("builder")))
    want = (row("L9", menu).get("fixed") or {}).get("gate_flips_mode")
    mode, _mc, src = DG.pinned_gate(ev, parent)
    if (mode or protocol("gate_flips_mode_default", menu)) == want:
        raise Defer("%s is already decided on flips_mode %s (%s): L9 would rebuild it unchanged (D15 gives card X6 "
                    "there)" % (parent, want, src))
    _once(ev, "L9", parent)
    replay, rc = DG._replay_mode(ev, parent)
    child = child_name(parent, ev.exps())
    params = {"exp": child, "replay_mode": replay}
    est, info, c = price("L9", params, ev, parent, menu)
    have = {_key(x) for x in d["cites"]}
    extra = [x for x in ([rc] if rc else []) + c if _key(x) not in have]
    return _proposal("L9", params, {}, d, d["cites"] + extra, est, info, parent, child, menu)


def _key(c):
    return json.dumps(c, sort_keys=True)


def _then_proposed(then, props):
    """True when a diagnosis's 'then' of the form '<lever> with <tokens>'
    (D2's 'L2 with --relevance', 'L2 with --increment-sources evidence') is
    already a proposal: a proposal of that lever whose argv holds those
    tokens in that order, side by side. Any other 'then' (a wait, a
    withheld lever, a card) stays deferred."""
    parts = str(then).split()
    if len(parts) < 3 or parts[1] != "with" or not parts[2].startswith("--"):
        return False
    want, n = parts[2:], len(parts) - 2
    return any(lid == parts[0] and any(list(av[i:i + n]) == want for i in range(len(av) - n + 1))
               for lid, av in props)


def propose(diagnoses, ev, menu=None):
    """Proposals, cards, operations, deferred and refused levers for the
    fired diagnoses. Identical commands from several diagnoses become one
    proposal with every trigger and cite; a card per off-menu lever."""
    menu = menu or load_menu()
    fired = {d["id"]: d for d in diagnoses if d.get("fired")}
    out = {"proposals": [], "cards": [], "operations": [], "deferred": [], "refused": []}
    props, cards, ops = {}, {}, {}
    thens = []

    def merge(target, d, cites):
        if d["id"] not in target["trigger"]:
            target["trigger"].append(d["id"])
        have = {_key(c) for c in target["cites"]}
        target["cites"] += [c for c in cites if _key(c) not in have]

    for d in [x for x in diagnoses if x.get("fired")]:
        for lid in d.get("levers") or []:
            if lid in menu.get("cards", {}):
                if lid in cards:
                    merge(cards[lid], d, d["cites"])
                    continue
                r = menu["cards"][lid]
                cards[lid] = M.card(lid, r["title"], [d["id"]], d["cites"], r.get("hypothesis", ""),
                                    r.get("why_menu_insufficient", ""), r.get("required_change", ""),
                                    r.get("cheapest_test", ""), r.get("control", ""),
                                    r.get("success_criterion", ""), _lit(r))
            elif lid in menu.get("operations", {}):
                r = menu["operations"][lid]
                k = (lid, d.get("exp"))
                if k in ops:
                    merge(ops[k], d, d["cites"])
                    continue
                op = {"op": lid, "title": r.get("title"), "risk": r["risk"], "kind": r.get("kind"),
                      "exp": d.get("exp") or ev.exp, "trigger": [d["id"]], "cites": list(d["cites"]),
                      "policy_action": r.get("policy_action"), "argv": None}
                if r.get("argv"):
                    op["argv"] = argv(lid, {"exp": op["exp"]}, {}, menu)
                ops[k] = op
            elif lid in menu.get("levers", {}):
                only = menu["levers"][lid].get("only_after") or []
                if only and not all(x in fired for x in only):
                    out["deferred"].append({"lever": lid, "trigger": d["id"],
                                            "reason": "only after %s" % only})
                    continue
                try:
                    built = _build(lid, d, ev, menu, fired)
                except Defer as e:
                    out["deferred"].append({"lever": lid, "trigger": d["id"], "reason": str(e)})
                    continue
                except LeverError as e:
                    out["refused"].append({"lever": lid, "trigger": d["id"], "reason": str(e)})
                    continue
                for p in built:
                    k = (lid, tuple(p["argv"]) if p["argv"] else (p.get("policy_action"),))
                    if k in props:
                        merge(props[k], d, p["cites"])
                    else:
                        props[k] = p
                        if p.get("waits_for"):
                            w = p["waits_for"]
                            out["deferred"].append({"lever": lid, "trigger": d["id"],
                                                    "reason": "then: %s after %s%s (%s is not on the cluster)"
                                                              % (lid, w.get("lever"),
                                                                 " --what %s" % w["what"] if w.get("what") else "",
                                                                 w.get("cluster_file"))})
            else:
                out["refused"].append({"lever": lid, "trigger": d["id"], "reason": "not on the menu"})
        for then in (d.get("detail") or {}).get("then") or []:
            out["deferred"].append({"lever": then.split()[0], "trigger": d["id"], "reason": "then: " + then})
            thens.append((out["deferred"][-1], then))
    # A 'then' the same tick already proposes (D2's 'L2 with --increment-sources
    # evidence' while D4's L2 carries that flag) is not also deferred.
    done = [id(x) for x, then in thens if _then_proposed(then, props)]
    out["deferred"] = [x for x in out["deferred"] if id(x) not in done]
    for did, lids in sorted((menu.get("supports") or {}).items()):
        if did in fired:
            for (lid, _), p in props.items():
                # support is evidence about one experiment: it joins a proposal
                # built on that same experiment only
                if lid in lids and fired[did].get("exp") == p.get("parent_exp"):
                    merge(p, fired[did], fired[did]["cites"])
    out["proposals"] = list(props.values())
    out["cards"] = list(cards.values())
    out["operations"] = list(ops.values())
    return out


def stable(result):
    """propose()'s result without its per-call ids and timestamps (for
    byte comparisons and the replay tests)."""
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k not in ("id", "created_utc")}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    return strip(result)
