"""Check a brain plan item by item before anything is filed (docs/INC_AUTOPILOT.md (c)).

The brain is a language model; its plan is a claim, not a decision. Each item
of a plan is checked on its own, and a failing item is DROPPED AND RECORDED
while the rest of the plan survives:

ranked_menu items (menu levers)
  * the lever is a row of the lever menu (brain_plan.load_menu: "menu" is
    "menu"); an X card (off-menu, R4) named here is dropped with a pointer to
    off_menu;
  * params are ones the lever row declares, agree with its "fixed" values, and
    pass `policy._check_params` against brain_plan.lever_bounds (the lever
    row's own param_bounds-form "params", else the policy row of its action;
    neither -> every params dict is refused);
  * the lever's preconditions hold on the fired diagnoses, the same rules
    levers.py applies: its "only_after" diagnoses fired, its "requires"
    params are given, and a real-loop build (brain_plan.D4_GATED_ACTIONS: L2,
    L5, L6) has D4 fired as decision_slot_ready (contract R4a);
  * at least one evidence cite, and every cite resolves against the snapshot
    (artifact, ledger line and/or JSON pointer) to EXACTLY the quoted value
    (same type, full precision; a bool is never a number);
  * every literature quote is a verbatim substring of the passage line it
    names in docs/literature and carries 20 characters of the passage's own
    text (corpus.Corpus.check_quote); a quote of a US line is marked a project
    note;
  * a well-formed `predicted` block and a falsifier;
  * no reference to test, ood22, ood23, imageweeds, a split manifest of one,
    or a holdout anywhere in the item: the text scan (LEAK_TEXT_RE) and the
    key scan the digest uses (brain_plan.dev_leaks: exam keys, score stamps
    and score paths, in cite values and params too). Literature quotes are
    exempt from the text scan: they are checked corpus text, not values of
    ours. The snapshot itself is scrubbed (brain_plan.dev_only) before any
    cite is resolved against it.

A ranked item that passes is then MATERIALISED the way levers.py builds a
deterministic proposal, so the brain chooses a lever and its free params and
nothing else:
  * derived values come from the evidence: the parent experiment, a new
    build's name when the plan gives none (levers.child_name, realloop_name),
    L2/L6's replay mode, recipes, base and relevance from a ready D4 and Step
    1, L5's definition from its parent loop (levers.loop_params, the flags
    levers.py rebuilds a loop with; levers.applied keeps a lever applied to a
    parent once), L4's manifest paths from
    exp.json, L8's manifest from Step 1. A plan may repeat a derived value
    only with the value the evidence gives (brain_plan.AUTOPILOT_SETS);
  * the price is computed by the levers.py estimators from the GPU time the
    parent measured (estimate_pilot_rebuild, estimate_realloop,
    estimate_baseline, estimate_walltime), plus, for a build, the build job's
    own walltime (levers.with_build_job, as levers.price adds it to the
    deterministic proposals). A plan's own est_gpu_hours is
    never used (kept on the proposal as `brain_est_gpu_hours`), and an item
    that cannot be priced from the evidence is dropped, never priced at 0;
  * the exact command is rendered with levers.argv (the lever's bounds and
    requires again), resolved and re-rendered by the executor's own
    resolve_params / render / argv_check, and authorised with
    policy.authorize for the tier2 actor: an item that actor could not file
    (L7, a missing policy row, a bound the rendered command breaks) is
    dropped with the reason;
  * the trigger is only what the brain names: a fired diagnosis with which
    the item shares at least one cite. With none the proposal is filed as
    `basis: "brain, no diagnosis"`;
  * a later item that renders the same request as an earlier valid one
    (brain_plan.proposal_key) is dropped as a duplicate.
A valid item becomes a model.proposal filed as `tier2:<model>`, carrying
argv, parent_exp, child_exp and the estimate with its cites.

off_menu items become R4 human research cards whose pre-registration draft
is built by `planner.make_experiment` (which refuses a draft with no control
or success criterion); their literature and any evidence cites are checked
the same way. R4 is never queued (approvals.propose refuses it): a card is
shown to a person, who decides whether it becomes a new protocol version.

Pure: reads the snapshot, the menu and the corpus it is given (and the job
scripts' #SBATCH --time through levers.estimate_walltime); writes nothing.
`outcome.record_validation()` puts the counts into the brain's track record.
"""
from __future__ import annotations

import math
import posixpath
import re

from . import brain_plan as BP
from . import model as M

VALIDATION_SCHEMA = "inc-plan-validation/1"

# Free-text references to a non-dev exam or the sealed split. Conservative on
# purpose: a false positive costs one dropped item, a miss costs the rule.
LEAK_TEXT_RE = re.compile(
    r"\b(?:ood22|ood23|imageweeds)\b"
    r"|\btest[ _-]?(?:split|set|exam|scores?|map|mAP50(?:-95)?|values?|results?|metrics?"
    r"|numbers?|json|jsonl|twelve|agnostic)\b"
    r"|\b(?:on|sealed|cwd12)\s+test\b"
    r"|\bh[oe]ld[ -]?out\b"
    r"|(?:^|[/\s\"'=:])test\.jsonl?\b"
    r"|scores/test|exams[./]test",
    re.I)

OFF_MENU_TEXT = ("hypothesis", "why_menu_insufficient", "required_change", "cheapest_test",
                 "control", "success_criterion")
RECIPES = ("full", "freeze", "lora")      # inc.pilot.inc_recipes(); predicted.chain
REALLOOP_BUILDER = "inc.realloop build"   # realloop.py exp.json "builder"
PILOT_BUILDER = "inc.pilot build"         # pilot.py exp.json "builder" (L9 rebuilds only a pilot)


class Refuse(Exception):
    """A ranked item that cannot be filed as the tier2 actor; the message says why."""


# --- helpers ------------------------------------------------------------------------

def _same(a, b):
    """Exact equality with JSON types: a bool is never a number, 1 == 1.0."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, str) or isinstance(b, str):
        return isinstance(a, str) and isinstance(b, str) and a == b
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    return False


def pointer_get(obj, pointer):
    """(ok, value or reason) for an RFC 6901 JSON pointer."""
    if pointer in (None, ""):
        return True, obj
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        return False, "pointer %r does not start with '/'" % (pointer,)
    for raw in pointer[1:].split("/"):
        tok = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(obj, dict):
            if tok not in obj:
                return False, "pointer %r: no key %r" % (pointer, tok)
            obj = obj[tok]
        elif isinstance(obj, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", tok) or int(tok) >= len(obj):
                return False, "pointer %r: no index %r" % (pointer, tok)
            obj = obj[int(tok)]
        else:
            return False, "pointer %r: %r is not a container" % (pointer, tok)
    return True, obj


def _cite_leak(cite):
    for field in ("artifact", "pointer"):
        v = cite.get(field)
        if isinstance(v, str):
            if BP._FORBIDDEN_PATH_RE.search(v):
                return "%s %r names a non-dev exam" % (field, v)
            segs = [s.replace("~1", "/").replace("~0", "~") for s in re.split(r"[/]", v)]
            bad = [s for s in segs if s in BP.FORBIDDEN_EXAMS or s.startswith("scores")
                   and any(e in s for e in BP.FORBIDDEN_EXAMS)]
            if bad:
                return "%s %r names a non-dev exam (%s)" % (field, v, bad[0])
    if "value" in cite:
        found = BP.dev_leaks(cite["value"], "/value")
        if found:
            return "the cited value carries non-dev exam data at %s" % ", ".join(found[:3])
    return ""


def resolve_cite(artifacts, cite):
    """(ok, reason). The cite must address a value in the snapshot equal to its own.

    `artifacts` is the dev-only map validate() builds (brain_plan.dev_only of
    the snapshot); a cite whose own value carries a non-dev exam key is
    refused whatever the snapshot holds."""
    if not isinstance(cite, dict):
        return False, "cite is not an object"
    leak = _cite_leak(cite)
    if leak:
        return False, "test leak: " + leak
    if "value" not in cite:
        return False, "cite carries no value"
    name = cite.get("artifact")
    if not isinstance(name, str) or name not in artifacts:
        return False, "artifact %r is not in the snapshot" % (name,)
    obj = artifacts[name]
    line, pointer = cite.get("line"), cite.get("pointer")
    if name.endswith(".jsonl"):
        if isinstance(line, bool) or not isinstance(line, int):
            return False, "a %s cite needs an integer line" % name
        if not 0 < line <= len(obj):
            return False, "%s has no line %d" % (name, line)
        obj = obj[line - 1]
    elif line is not None:
        return False, "%s is not line-addressed; use a pointer" % name
    elif pointer in (None, ""):
        return False, "a %s cite needs a JSON pointer" % name
    ok, got = pointer_get(obj, pointer)
    if not ok:
        return False, got
    if not _same(got, cite["value"]):
        return False, "%s%s%s holds %s, the cite says %s" % (
            name, "" if line is None else ":%d" % line, pointer or "",
            _short(got), _short(cite["value"]))
    return True, ""


def _short(v, n=80):
    s = repr(v)
    return s if len(s) <= n else s[:n] + "..."


def _text_leaks(obj, path=""):
    """[(path, phrase)] of leak phrases in every string (and dict key) of obj."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            m = LEAK_TEXT_RE.search(str(k))
            if m:
                found.append(("%s/%s" % (path, k), m.group(0)))
            found.extend(_text_leaks(v, "%s/%s" % (path, k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found.extend(_text_leaks(v, "%s/%d" % (path, i)))
    elif isinstance(obj, str):
        m = LEAK_TEXT_RE.search(obj)
        if m:
            found.append((path, m.group(0).strip()))
    return found


def _leaks_outside_lit(item):
    """[(path, what)]: the text scan, the digest's key scan (exam keys, score
    stamps, score paths) and the cite address scan, over everything but the
    literature quotes."""
    scan = {k: v for k, v in item.items() if k != "lit_cites"}
    leaks = _text_leaks(scan)
    leaks.extend((p, "a non-dev exam key, score stamp or score path")
                 for p in BP.dev_leaks(scan))
    for i, c in enumerate(item.get("evidence_cites") or []):
        if isinstance(c, dict):
            why = _cite_leak(c)
            if why:
                leaks.append(("/evidence_cites/%d" % i, why))
    return leaks


def _check_lit(corpus, lits, reasons, counts):
    if lits is None:
        return []
    if not isinstance(lits, list):
        reasons.append("lit_cites is not a list")
        return []
    ok_lits = []
    for i, lc in enumerate(lits):
        counts["lit_checked"] += 1
        if not isinstance(lc, dict):
            counts["lit_failed"] += 1
            reasons.append("lit_cites[%d] is not an object" % i)
            continue
        if corpus is None:
            counts["lit_failed"] += 1
            reasons.append("lit_cites[%d]: no literature corpus to check against" % i)
            continue
        ok, why = corpus.check_quote(lc.get("paper_id"), lc.get("line"), lc.get("quote"))
        if not ok:
            counts["lit_failed"] += 1
            reasons.append("lit_cites[%d]: %s" % (i, why))
            continue
        kind = (corpus.passage(lc["paper_id"], lc["line"]) or {}).get("kind")
        rec = {"paper_id": corpus.canonical_id(lc["paper_id"]), "line": lc["line"],
               "quote": lc["quote"], "kind": kind}
        if kind == "US":
            # A US line is this project's reading of the paper, not its finding.
            rec["project_note"] = True
            counts["lit_project_notes"] += 1
        ok_lits.append(rec)
    return ok_lits


def _check_evidence(artifacts, cites, reasons, counts, required):
    if cites is None:
        cites = []
    if not isinstance(cites, list):
        reasons.append("evidence_cites is not a list")
        return []
    if required and not cites:
        reasons.append("no evidence cite: a proposal that cites nothing is not a diagnosis")
    good = []
    for i, c in enumerate(cites):
        counts["cites_checked"] += 1
        ok, why = resolve_cite(artifacts, c)
        if not ok:
            counts["cites_failed"] += 1
            reasons.append("evidence_cites[%d]: %s" % (i, why))
            continue
        good.append({"artifact": c["artifact"], "line": c.get("line"),
                     "pointer": c.get("pointer"), "value": c["value"]})
    return good


def _check_predicted(pred, reasons):
    if not isinstance(pred, dict):
        reasons.append("predicted is missing or not an object")
        return None
    metric, direction, mag = pred.get("metric"), pred.get("direction"), pred.get("magnitude")
    chain = pred.get("chain")
    if metric not in BP.PREDICT_METRICS:
        reasons.append("predicted.metric %r is not one of %s" % (metric, BP.PREDICT_METRICS))
    if direction not in BP.PREDICT_DIRECTIONS:
        reasons.append("predicted.direction %r is not one of %s"
                       % (direction, BP.PREDICT_DIRECTIONS))
    if mag is not None and (isinstance(mag, bool) or not isinstance(mag, (int, float))
                            or not math.isfinite(mag) or mag < 0):
        reasons.append("predicted.magnitude %r is not null or a finite number >= 0" % (mag,))
    if chain is not None and chain not in RECIPES:
        reasons.append("predicted.chain %r is not null or one of %s" % (chain, RECIPES))
    out = {"metric": metric, "direction": direction, "magnitude": mag}
    if chain is not None:
        out["chain"] = chain
    return out


def _new_counts():
    return {"items": 0, "valid": 0, "cards": 0, "dropped": 0, "cites_checked": 0,
            "cites_failed": 0, "lit_checked": 0, "lit_failed": 0, "lit_project_notes": 0,
            "leaks": 0, "param_refusals": 0, "precondition_refusals": 0, "unpriced": 0,
            "render_refusals": 0, "duplicates": 0, "no_diagnosis": 0,
            "off_menu_levers_in_ranked": 0, "unknown_levers": 0}


# --- preconditions and materialisation ------------------------------------------------

def preconditions(lever, row, params, ctx):
    """[reasons] a menu lever may not be proposed now: the rules levers.py
    applies to its own proposals (only_after, requires) and the D4 gate on
    real-loop builds (contract R4a)."""
    reasons = []
    fired = ctx["diags"]
    only = [d for d in row.get("only_after") or []]
    missing = [d for d in only if d not in fired]
    if missing:
        reasons.append("lever %s runs only after %s fired; %s did not fire on this evidence"
                       % (lever, ", ".join(only), ", ".join(missing)))
    need = [k for k in row.get("requires") or [] if params.get(k) is None]
    if need:
        reasons.append("lever %s requires the param(s) %s" % (lever, ", ".join(need)))
    if row.get("policy_action") in BP.D4_GATED_ACTIONS:
        d4 = fired.get("D4")
        if not d4 or d4.get("name") != BP.D4_READY:
            reasons.append("lever %s builds the real loop, which needs D4 fired as %s (no real loop "
                           "while no recipe tracks the truth arm); D4 is %s on this evidence"
                           % (lever, BP.D4_READY, (d4 or {}).get("name") or "silent"))
    return reasons


def _check_derived(bp, derived, what):
    for k, v in sorted(derived.items()):
        if bp.get(k) is not None and not _same(bp[k], v):
            raise Refuse("%s=%s is not the plan's to choose: the autopilot takes it from %s (%s)"
                         % (k, _short(bp[k]), what, _short(v)))


def _new_name(name, exps):
    if name in exps:
        raise Refuse("experiment %r already exists in the evidence; an experiment is built once"
                     % name)
    return name


def materialise(lever, row, bp, ctx):
    """{"params", "derived", "parent_exp", "child_exp", "est", "estimate", "est_cites"}.

    The brain's free params `bp` (its price already set aside) turned into
    the request levers.py would build: derived values from the evidence, a
    new build's name, and the price from the levers.py estimators. Raises
    levers.Defer when the evidence cannot give a value (never guessed) and
    Refuse when the plan contradicts it.
    """
    from . import levers as L
    ev, menu = ctx["ev"], ctx["lmenu"]
    exps = ev.exps()
    parent = ctx["exp"]
    derived, child, est_cites = {}, None, []
    params = dict(bp)
    if lever == "L1":
        if ev.json("%s/exp.json" % parent) is None:
            raise L.Defer("no %s/exp.json in the evidence to rebuild" % parent)
        gp, gc = L.gate_params(ev, parent, menu)
        _check_derived(bp, {"gate_flips_mode": gp.get("gate_flips_mode")},
                       "the parent pilot %s's pinned gate (L9 is the lever that changes it)" % parent)
        child = _new_name(bp.get("exp") or L.child_name(parent, exps), exps)
        params["exp"] = child
        params.pop("gate_flips_mode", None)
        params.update(gp)
        params.update(row.get("fixed") or {})
        est, info, est_cites = L.estimate_pilot_rebuild(ev, parent, params.get("replay_mode", "full"))
        est_cites = list(gc) + list(est_cites)
    elif lever == "L9":
        pd = ev.json("%s/exp.json" % parent)
        if pd is None:
            raise L.Defer("no %s/exp.json in the evidence to rebuild" % parent)
        if pd.get("builder") != PILOT_BUILDER:
            raise Refuse("L9 rebuilds a pilot; %s is not one (builder %r)" % (parent, pd.get("builder")))
        from . import diagnose as DG
        want = (row.get("fixed") or {}).get("gate_flips_mode")
        mode, _mc, src = DG.pinned_gate(ev, parent)
        if (mode or L.protocol("gate_flips_mode_default", menu)) == want:
            raise Refuse("%s is already decided on flips_mode %s (%s): L9 would rebuild it unchanged"
                         % (parent, want, src))
        replay, rc = DG._replay_mode(ev, parent)
        _check_derived(bp, {"replay_mode": replay}, "the parent pilot %s (L9 changes the gate alone)" % parent)
        L._once(ev, "L9", parent)
        child = _new_name(bp.get("exp") or L.child_name(parent, exps), exps)
        params = {"exp": child, "replay_mode": replay}
        params.update(row.get("fixed") or {})
        est, info, est_cites = L.estimate_pilot_rebuild(ev, parent, replay)
        est_cites = ([rc] if rc else []) + list(est_cites)
    elif lever in ("L2", "L6"):
        d4 = ctx["diags"].get("D4")
        if not d4 or d4.get("name") != BP.D4_READY:
            raise Refuse("lever %s takes its replay mode and recipes from D4 fired as %s"
                         % (lever, BP.D4_READY))
        # The relevance criterion (levers.increment_criterion) is checked at the
        # plan's own --size and --n-verified: an evidence build must fit them.
        # A plan that names neither gets the deterministic L2's: the defaults,
        # or N and M sized by the R4 rule (levers.evidence_sizing), which `base`
        # then carries as --size / --n-verified.
        base, parent, crit_cites = L._l2_params(ev, d4, menu, size=bp.get("size"),
                                                n_verified=bp.get("n_verified"))
        _check_derived(bp, {k: base.get(k) for k in ("base", "relevance", "increment_sources", "replay_mode",
                                                     "recipes", "gate_flips_mode")},
                       "D4 on %s and Step 1" % parent)
        child = _new_name(bp.get("exp") or base["exp"], exps)
        params = dict(base, exp=child)
        for k in ("size", "n_verified"):
            if bp.get(k) is not None:
                params[k] = bp[k]
        params.update(row.get("fixed") or {})
        est, info, est_cites = L.estimate_realloop(ev, params, menu)
        est_cites = list(crit_cites) + list(est_cites)
    elif lever == "L5":
        pd = ev.json("%s/exp.json" % parent)
        if pd is None:
            raise L.Defer("no %s/exp.json in the evidence" % parent)
        if pd.get("builder") != REALLOOP_BUILDER:
            raise Refuse("L5 rebuilds a real loop with a larger --size; %s is not one (builder %r)"
                         % (parent, pd.get("builder")))
        # The parent loop's own build flags, read the way levers.py reads them for
        # its own L5 (--base is Step 1's source manifest, --n-verified from
        # build_summary.json, --no-truth when the parent had no truth arm), so a
        # brain L5 and a deterministic L5 of the same parent are one request.
        own, own_cites = L.loop_params(ev, parent)
        _check_derived(bp, dict({k: v for k, v in own.items() if k != "size"},
                                gate_flips_mode=own.get("gate_flips_mode"),
                                increment_sources=own.get("increment_sources"), relevance=own.get("relevance")),
                       "the parent loop %s" % parent)
        if bp.get("no_truth") is not None and "no_truth" not in own:
            raise Refuse("L5 keeps the parent loop's truth arm; dropping it (--no-truth) is L6")
        if bp.get("size") is None:
            raise Refuse("lever L5 requires the param size")
        cur = own.get("size")
        if isinstance(cur, int) and not isinstance(cur, bool) and bp["size"] <= cur:
            raise Refuse("L5 is a larger --size: %d is not above %s's %d" % (bp["size"], parent, cur))
        params = dict(own, size=bp["size"])
        # Lineage (levers.applied): a lever is applied to a parent once, whether
        # the earlier application is a later loop in the evidence or a build the
        # ticker has in flight (the context's lineage records).
        L._once(ev, "L5", parent, params=params)
        crit_cites = L.check_criterion(ev, params, L.inc_dir(ev, parent))
        child = _new_name(bp.get("exp") or L.child_name(parent, exps), exps)
        params["exp"] = child
        est, info, est_cites = L.estimate_realloop(ev, params, menu)
        est_cites = list(own_cites) + list(crit_cites) + list(est_cites)
    elif lever == "L3":
        L._l3_ok(ev)
        if not row.get("script"):
            raise L.Defer("lever L3 names no job script to read its walltime from")
        parent = None
        est, info = L.estimate_walltime(row["script"])
    elif lever == "L4":
        exp = bp.get("exp") or parent
        derived, est_cites = L.audit_args(ev, exp)
        _check_derived(bp, {"trusted": derived["trusted"], "audits": derived["audits"],
                            "out": derived["out"]}, "%s/exp.json" % exp)
        params = {"exp": exp}
        parent = exp
        if not row.get("script"):
            raise L.Defer("lever L4 names no job script to read its walltime from")
        est, info = L.estimate_walltime(row["script"])
    elif lever == "L7":
        params = {k: bp[k] for k in ("exp", "unit", "cause") if k in bp}
        parent = bp.get("exp")
        est, info = 0.0, {"estimator": "zero"}
    elif lever == "L8":
        child = _new_name(bp.get("exp") or L.BASELINE_B_EXP, exps)
        manifest = posixpath.join(L.inc_dir(ev), "step1", "base_B.jsonl")
        _check_derived(bp, {"manifest": manifest}, "Step 1 (INC_DIR/step1/base_B.jsonl)")
        sel = ev.json("step1/select_summary.json")
        n = ((sel or {}).get("sizes") or {}).get("base_B")
        if isinstance(n, bool) or not isinstance(n, (int, float)) or not n:
            raise L.Defer("no Step 1 select build (sizes.base_B) in the evidence to price L8")
        params = {"exp": child, "manifest": manifest}
        parent = None
        est, info, est_cites = L.estimate_baseline(ev, n, menu)
        est_cites = est_cites + [ev.cite("step1/select_summary.json", "/sizes/base_B")]
    else:
        raise Refuse("the autopilot has no builder for lever %s" % lever)
    if row.get("policy_action", "").startswith("inc_build_"):
        # The job that builds the experiment is charged too (levers.price does
        # the same for the deterministic proposals).
        est, info = L.with_build_job(est, info)
    return {"params": params, "derived": derived, "parent_exp": parent, "child_exp": child,
            "est": float(est), "estimate": info, "est_cites": list(est_cites)}


def file_check(lever, action, params, derived, est, ctx):
    """(argv, policy check) of the materialised request, or Refuse.

    The command is levers.argv's (the lever's bounds and requires); the
    executor's resolve_params / render / argv_check read it back into the
    policy params it would run, and policy.authorize must allow the tier2
    actor to file exactly those."""
    from . import executor as X
    from . import levers as L
    from ..brain import policy as POL
    try:
        argv = L.argv(lever, params, derived, ctx["lmenu"])
    except L.LeverError as e:
        raise Refuse(str(e))
    row = POL.describe(action)
    if not row.get("known"):
        raise Refuse("the policy table has no row %r: %s" % (action, row.get("reason")))
    try:
        policy_params, _meta = X.resolve_params(action, row, params, argv, est)
        rendered = X.render(action, policy_params)
    except X.ExecError as e:
        raise Refuse("the executor cannot run it: %s" % e)
    ok, why = X.argv_check(rendered, argv)
    if not ok:
        raise Refuse(why)
    auth = POL.authorize(ctx["actor"], action, policy_params)
    if not auth.get("allowed"):
        raise Refuse("%s could not file it: %s" % (ctx["actor"], "; ".join(auth.get("reasons") or [])))
    cost = POL.estimate_su(action, policy_params)
    f = row.get("est_su") or {}
    if "gpu_type" in f and cost.get("su") is None:
        raise Refuse("%s costs GPU time and its price is unknown (%s)" % (action, cost.get("reason")))
    return argv, {"authorize": list(auth.get("reasons") or []),
                  "needs_approval": bool(auth.get("needs_approval")),
                  "est_su": cost.get("su"), "policy_params": policy_params}


def _cite_key(c):
    return (c.get("artifact"), c.get("line"), c.get("pointer"))


def trigger_of(item, cites, ctx):
    """(trigger ids, [unsupported]): a diagnosis the item names counts only when
    it fired and the item repeats one of its cites."""
    named = item.get("trigger")
    if not isinstance(named, list):
        named = []
    have = {_cite_key(c) for c in cites}
    out, bad = [], []
    for t in named:
        d = ctx["diags"].get(t) if isinstance(t, str) else None
        if d is None:
            bad.append({"id": t, "why": "did not fire on this evidence"})
        elif not have & {_cite_key(c) for c in d.get("cites") or [] if isinstance(c, dict)}:
            bad.append({"id": t, "why": "the item repeats none of its cites"})
        elif t not in out:
            out.append(t)
    return out, bad


# --- items --------------------------------------------------------------------------

def check_ranked(item, rank, ctx, counts):
    """(proposal or None, reasons)."""
    reasons = []
    if not isinstance(item, dict):
        return None, ["item is not an object"]
    leaks = _leaks_outside_lit(item)
    if leaks:
        counts["leaks"] += 1
        return None, ["test leak: %s at %s" % (phrase, path or "/") for path, phrase in leaks[:5]]
    lever = item.get("lever")
    row = ctx["menu"].get(lever) if isinstance(lever, str) else None
    if row is None:
        counts["unknown_levers"] += 1
        return None, ["lever %r is not on the menu; an idea outside it belongs in off_menu"
                      % (lever,)]
    if row.get("menu") != "menu":
        counts["off_menu_levers_in_ranked"] += 1
        return None, ["lever %s is off-menu (%s): it cannot be queued; state it under "
                      "off_menu with a control and a success criterion" % (lever,
                                                                          row.get("risk") or "R4")]
    params = item.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return None, ["params is not an object"]
    params = dict(params)
    # The price is the autopilot's (contract (b), "Cost"): a plan's own figure
    # is kept for the record and never used.
    brain_price = params.pop(BP.PRICE_PARAM, None)
    param_reasons = []
    fixed = row.get("fixed_params") or {}
    for k, v in fixed.items():
        if k in params and not _same(params[k], v):
            param_reasons.append("param %r=%s contradicts the lever's fixed value %s"
                                 % (k, _short(params[k]), _short(v)))
    # A lever takes only the params its own row declares: L1's argv fixes
    # --replay-mode full, so a recipes param on L1 would be a different lever
    # wearing L1's name even where the policy row would accept it.
    declared = BP.declared_bounds(row)
    if declared:
        for k in params:
            if k not in declared and k not in fixed:
                param_reasons.append("lever %s takes no param %r (it declares %s)"
                                     % (lever, k, ", ".join(sorted(declared))))
    final = dict(fixed)
    final.update(params)
    action = row.get("policy_action")
    bounds, source = BP.lever_bounds(row)
    if bounds is None:
        param_reasons.append("%s; params refused" % source)
    else:
        from ..brain import policy
        ok, why = policy._check_params(final, bounds)
        if not ok:
            param_reasons.extend("params: %s" % w for w in why)
    from . import levers as L
    if action == "inc_build_realloop" and L.evidence_with_relevance(final):
        param_reasons.append("params: %s" % M.EVIDENCE_WITH_RELEVANCE)
    if param_reasons:
        counts["param_refusals"] += 1
        reasons.extend(param_reasons)
    pre = preconditions(lever, row, params, ctx)
    if pre:
        counts["precondition_refusals"] += 1
        reasons.extend(pre)
    cites = _check_evidence(ctx["artifacts"], item.get("evidence_cites"), reasons, counts,
                            required=True)
    lits = _check_lit(ctx["corpus"], item.get("lit_cites"), reasons, counts)
    pred = _check_predicted(item.get("predicted"), reasons)
    for f in ("rationale", "falsifier"):
        if not isinstance(item.get(f), str) or not item[f].strip():
            reasons.append("%s is missing" % f)
    if reasons:
        return None, reasons
    from . import levers as L
    try:
        mat = materialise(lever, row, params, ctx)
    except L.Defer as e:
        counts["unpriced"] += 1
        return None, ["deferred: %s; the autopilot prices and derives every item from the evidence "
                      "and never guesses" % e]
    except (Refuse, L.LeverError) as e:
        counts["render_refusals"] += 1
        return None, [str(e)]
    except Exception as e:           # one item's failure never takes the plan down
        counts["render_refusals"] += 1
        return None, ["could not be materialised from the evidence (%s: %s)"
                      % (type(e).__name__, e)]
    req = dict(mat["params"])
    req[BP.PRICE_PARAM] = mat["est"]
    try:
        argv, check = file_check(lever, action, req, mat["derived"], mat["est"], ctx)
    except Refuse as e:
        counts["render_refusals"] += 1
        return None, [str(e)]
    except Exception as e:
        counts["render_refusals"] += 1
        return None, ["could not be rendered for filing (%s: %s)" % (type(e).__name__, e)]
    risk = row.get("risk")
    if risk not in M.RISKS:
        from ..brain import policy
        risk = policy.risk_of(action)
    trigger, unsupported = trigger_of(item, cites, ctx)
    if not trigger:
        counts["no_diagnosis"] += 1
    prop = M.proposal(lever, argv, req, action, risk, trigger, cites, lit=lits,
                      control=str(row.get("control") or ""),
                      success=str(row.get("success") or row.get("success_criterion") or ""),
                      falsifier=item["falsifier"].strip(), est_gpu_hours=mat["est"],
                      proposed_by=ctx["actor"])
    # Additive fields: the brain's rank, its reasoning and the prediction
    # outcome.py scores; lineage and the estimate as levers.py records them.
    prop.update({"rank": rank, "rationale": item["rationale"].strip(), "predicted": pred,
                 "basis": "diagnosis" if trigger else "brain, no diagnosis",
                 "trigger_unsupported": unsupported,
                 "parent_exp": mat["parent_exp"], "child_exp": mat["child_exp"],
                 "via": row.get("via"), "title": row.get("title"),
                 "estimate": dict(mat["estimate"], cites=mat["est_cites"]),
                 "brain_est_gpu_hours": brain_price, "policy_check": check})
    return prop, []


def check_off_menu(item, ctx, counts):
    """(R4 card or None, reasons)."""
    if not isinstance(item, dict):
        return None, ["item is not an object"]
    leaks = _leaks_outside_lit(item)
    if leaks:
        counts["leaks"] += 1
        return None, ["test leak: %s at %s" % (phrase, path or "/") for path, phrase in leaks[:5]]
    reasons = []
    for f in ("hypothesis", "why_menu_insufficient", "required_change"):
        if not isinstance(item.get(f), str) or not item[f].strip():
            reasons.append("%s is missing" % f)
    lits = _check_lit(ctx["corpus"], item.get("lit_cites"), reasons, counts)
    cites = _check_evidence(ctx["artifacts"], item.get("evidence_cites"), reasons, counts,
                            required=False)
    if reasons:
        return None, reasons
    from ..brain import planner
    made = planner.make_experiment(
        recipe="off-menu: %s" % item["required_change"].strip(), params={},
        control=item.get("control"), risk="R4",
        success_criterion=item.get("success_criterion"),
        stop_rule=str(item.get("cheapest_test") or "").strip() and
        "the cheapest test (%s) refutes the hypothesis" % item["cheapest_test"].strip())
    if not made.get("ok"):
        return None, ["planner.make_experiment refused the draft: %s" % made.get("reason")]
    trigger, unsupported = trigger_of(item, cites, ctx)
    card = M.card(str(item.get("lever") or "off_menu"),
                  str(item.get("title") or item["hypothesis"].strip()[:120]), trigger, cites,
                  hypothesis=item["hypothesis"].strip(),
                  why_menu_insufficient=item["why_menu_insufficient"].strip(),
                  required_change=item["required_change"].strip(),
                  cheapest_test=str(item.get("cheapest_test") or "").strip(),
                  control=str(item.get("control") or "").strip(),
                  success_criterion=str(item.get("success_criterion") or "").strip(),
                  lit=lits, proposed_by=ctx["actor"])
    # Additive: the pre-registration draft, and why a card is never queued.
    card.update({"kind": "research_card", "status": "draft", "experiment": made["experiment"],
                 "basis": "diagnosis" if trigger else "brain, no diagnosis",
                 "trigger_unsupported": unsupported,
                 "note": "R4 is never queued (approvals.propose refuses it). A person decides "
                         "whether this becomes a new protocol version; only then does it join "
                         "the lever menu with its literature and a replay case."})
    return card, []


def _stop(rec, counts, dropped):
    if rec is None:
        return None
    if isinstance(rec, str):
        rec = {"stop": None, "reason": rec}
    if not isinstance(rec, dict):
        dropped.append({"section": "stop_recommendation", "index": 0,
                        "reasons": ["not an object or string"]})
        return None
    leaks = _text_leaks(rec) + [(p, "a non-dev exam key") for p in BP.dev_leaks(rec)]
    if leaks:
        counts["leaks"] += 1
        dropped.append({"section": "stop_recommendation", "index": 0,
                        "reasons": ["test leak: %s at %s" % (phrase, path or "/")
                                    for path, phrase in leaks[:5]]})
        return None
    stop = rec.get("stop")
    return {"stop": stop if isinstance(stop, bool) else None,
            "reason": str(rec.get("reason") or "")[:2000]}


# --- the snapshot as evidence -----------------------------------------------------------

def evidence_of(snapshot, exp=None):
    """An evidence.Evidence of the snapshot (the levers.py estimators and
    derivations read one): the Evidence itself, or one rebuilt from an
    artifacts map (brain_plan.artifacts_of's shape), scrubbed dev-only."""
    from . import evidence as EV
    if all(hasattr(snapshot, a) for a in ("json", "cite", "exps", "artifacts", "ledgers")):
        return snapshot
    arts = BP.dev_only(BP.artifacts_of(snapshot) or {})
    ev = EV.Evidence(exp or "unknown")
    for name, obj in sorted(arts.items()):
        if name.endswith("/ledger.jsonl") and isinstance(obj, list):
            ev.ledgers[name[:-len("/ledger.jsonl")]] = [(i + 1, e) for i, e in enumerate(obj)
                                                        if e is not None]
        else:
            ev.artifacts[name] = obj
    if exp is None:
        names = ev.exps()
        ev.exp = names[-1] if names else None
    return ev


def _levers_menu(menu):
    """The lever rows in levers.json's shape (levers.argv and the estimators
    read that shape), with the builders' protocol constants from levers.json."""
    from . import levers as L
    try:
        protocol = L.load_menu().get("protocol") or {}
    except OSError:
        protocol = {}
    return {"levers": {k: r for k, r in menu.items() if r.get("menu") == "menu"},
            "cards": {k: r for k, r in menu.items() if r.get("menu") != "menu"},
            "operations": {}, "protocol": protocol}


def validate(reply, artifacts, menu, corpus, model=None, diagnoses=None, exp=None):
    """The validation record of one plan.

    `reply` is a brain_plan reply ({"schema": inc-plan-reply/1, "plan": ...})
    or a bare plan dict. `artifacts` is the dev-only snapshot the digest was
    built from (an evidence.Evidence, or brain_plan.artifacts_of's map; it is
    scrubbed again here); `menu` the lever rows (brain_plan.load_menu);
    `corpus` a corpus.Corpus (None fails every literature cite); `diagnoses`
    the diagnoses of that snapshot (the fired ones decide preconditions and
    triggers; None = nothing fired); `exp` the experiment the plan is for
    (default: the reply's, else the snapshot's latest), the parent of a
    rebuild.
    """
    plan = reply.get("plan") if isinstance(reply, dict) and "schema" in reply else reply
    model = model or (reply.get("model") if isinstance(reply, dict) else None) or "unknown"
    out = {"schema": VALIDATION_SCHEMA, "ok": isinstance(plan, dict), "model": model,
           "proposed_by": BP.actor_for(model), "validated_utc": M.utc_now(),
           "digest_sha256": reply.get("digest_sha256") if isinstance(reply, dict) else None,
           "menu": [], "cards": [], "dropped": [], "stop_recommendation": None,
           "counts": _new_counts()}
    if not isinstance(plan, dict):
        out["reason"] = "no plan to validate"
        return out
    counts = out["counts"]
    exp = exp or (reply.get("exp") if isinstance(reply, dict) and "schema" in reply else None)
    ev = evidence_of(artifacts, exp)
    exp = exp or ev.exp
    diags = {d.get("id"): d for d in (diagnoses or []) if isinstance(d, dict) and d.get("fired")}
    ctx = {"artifacts": BP.dev_only(BP.artifacts_of(artifacts) or {}), "menu": menu or {},
           "corpus": corpus, "actor": out["proposed_by"], "diags": diags,
           "fired": {k: list(d.get("levers") or []) for k, d in diags.items()},
           "ev": ev, "exp": exp, "lmenu": _levers_menu(menu or {})}
    out["exp"] = exp
    seen = {}
    for i, item in enumerate(plan.get("ranked_menu") or []):
        counts["items"] += 1
        prop, reasons = check_ranked(item, i + 1, ctx, counts)
        if prop is not None:
            key = BP.proposal_key(prop)
            if key in seen:
                counts["duplicates"] += 1
                prop, reasons = None, ["duplicate of ranked item %d: the same request (%s)"
                                       % (seen[key], " ".join(prop["argv"]))]
            else:
                seen[key] = i + 1
        if prop is None:
            counts["dropped"] += 1
            out["dropped"].append({"section": "ranked_menu", "index": i,
                                   "lever": item.get("lever") if isinstance(item, dict) else None,
                                   "reasons": reasons})
        else:
            counts["valid"] += 1
            out["menu"].append(prop)
    for i, item in enumerate(plan.get("off_menu") or []):
        counts["items"] += 1
        card, reasons = check_off_menu(item, ctx, counts)
        if card is None:
            counts["dropped"] += 1
            out["dropped"].append({"section": "off_menu", "index": i, "reasons": reasons})
        else:
            counts["cards"] += 1
            out["cards"].append(card)
    out["stop_recommendation"] = _stop(plan.get("stop_recommendation"), counts, out["dropped"])
    return out
