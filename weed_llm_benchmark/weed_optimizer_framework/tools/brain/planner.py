"""The campaign planner: one digest in, one Plan out (v3.29.0, WP6).

Why this exists
----------------
`digest.py` compresses a whole campaign into one call's worth of context.
Nothing before this module turned that compressed history into a concrete
next move. A "next move" that is worth anything to this campaign has to name
what it expects, what it will run, what it measures that against, and when it
stops -- a plan that proposes changing something with no stated control
measures the campaign's own drift, and one with no stated success criterion
can never conclude. Both are refused at the one place every backend below
constructs an experiment entry (`make_experiment`), so no backend -- including
a future one -- can smuggle either past `validate()` later.

Backends
--------
* `mock` -- deterministic, rule-based, reads only the digest's own rendered
  `text`/`sections`. No model, no network, no clock. Always marked
  `simulated: True`, because a fabricated stand-in presented as a model's
  plan would misrepresent the result the campaign later reports.
* `file` -- the same deterministic rules as `mock`, additionally persisted as
  a new, never-overwritten version under
  `results/framework/_brain/<domain>/plans/`. Still `simulated: True`: the
  content is the same rule engine, only the storage differs.
* `cluster:<model>` -- one call through `supervisor.OpenAICompatClient` (or
  an injected callable of the same shape), the reply's JSON schema-checked
  and each proposed experiment run through `make_experiment`. An unreachable
  endpoint, a reply with no parseable JSON, or a reply that is not an object
  is a REFUSED plan (`refused: True`, a stated `reason`, empty lists) --
  never a plan that merely looks empty, which a caller could otherwise read
  as "the model looked and found nothing to propose."

The lever blindfold
--------------------
`TIERED_SUPERVISION_PLAN.md` states the evaluation design: a planner under
evaluation is handed a digest built with `omit_levers=True`, specifically so
a planner cannot do answer-key lookup against the pre-registered menu instead
of reasoning about the campaign's own symptoms. This module enforces that
blindfold itself rather than trusting every caller to withhold a `lever_menu`
opt correctly: `_levers_included()` inspects the digest's OWN `sections`
metadata (the same list `digest.build()` returns) for a `levers` entry with
`included: True`, and a caller-supplied `lever_menu` is used only when that
is found. A digest built with `omit_levers=True` never carries that entry, so
a lever_menu passed anyway is silently ignored -- a defect in the calling
code (unaligning which digest was built for which arm) does not become a
leaked answer key.

`validate(plan)` is structural only, mirroring `supervisor.check_shape()`'s
`(ok, problems)` contract: it can say a plan is well-formed, never that it is
a good plan. That judgement, and whether a completed experiment's result
actually beat its control, is `experiments.verdict()`'s job once real numbers
land.

Pure stdlib. No Mongo import, no network call except through an injected or
default `OpenAICompatClient`, so this module imports as freely inside a SLURM
job body as it does on the always-on dashboard process.
"""
import json
import pathlib
import re

from . import corpus
from . import parallelism
from . import policy
from . import supervisor

TOOL_VERSION = "wp6-planner/1"

BACKENDS = ("mock", "file")  # plus any "cluster:<model>" string


# --- experiment construction: the one gate every backend passes through ----
def make_experiment(recipe, params=None, seeds=None, control="", est_su=None,
                    risk=None, success_criterion="", stop_rule=""):
    """{"ok": True, "experiment": {...}} or {"ok": False, "reason": str}.

    The only place an `ordered_experiments` entry is assembled. A missing
    `control` means the run has nothing to measure drift against; a missing
    `success_criterion` means the run has no way to conclude. Both refuse
    here, with the reason, rather than downstream in `validate()` where the
    specific offending backend would already be lost.
    """
    recipe = str(recipe or "").strip()
    control = str(control or "").strip()
    success_criterion = str(success_criterion or "").strip()
    if not recipe:
        return {"ok": False, "reason": "experiment names no recipe"}
    if not control:
        return {"ok": False, "reason": (
            "experiment %r names no control; a run with no control measures "
            "the campaign's own drift, not the thing under test" % (recipe,))}
    if not success_criterion:
        return {"ok": False, "reason": (
            "experiment %r names no success_criterion; a run with no "
            "criterion has no way to conclude" % (recipe,))}
    if risk is not None and risk not in policy.RISKS:
        return {"ok": False, "reason": (
            "experiment %r risk %r is not one of %s" % (recipe, risk, policy.RISKS))}
    if est_su is not None and (isinstance(est_su, bool) or not isinstance(est_su, (int, float))):
        return {"ok": False, "reason": (
            "experiment %r est_su %r is not a number or null" % (recipe, est_su))}
    exp = {
        "recipe": recipe,
        "params": dict(params) if isinstance(params, dict) else {},
        "seeds": list(seeds) if isinstance(seeds, list) else [],
        "control": control,
        "est_su": est_su,
        "risk": risk,
        "success_criterion": success_criterion,
        "stop_rule": str(stop_rule or "").strip() or
            "stop if any seed's run does not complete",
    }
    return {"ok": True, "experiment": exp}


# --- reading what the digest itself shows -----------------------------------
def _levers_included(sections):
    """True only if the DIGEST's own metadata says its levers section shipped.

    Deliberately does not ask the caller. See the module docstring's "lever
    blindfold" section: trusting a caller-supplied flag here is exactly the
    gap that would let a misaligned caller leak the sealed menu into the
    evaluated arm.
    """
    for s in (sections or []):
        if isinstance(s, dict) and s.get("name") == "levers" and s.get("included") is True:
            return True
    return False


_TREND_RE = re.compile(r"OPEN QUESTION: given that (.+?) and (\d+) unresolved signal")
_SIGNAL_LINE_RE = re.compile(r"^\s*-\s+(\S+)\s+\((info|warn|crit)\):\s*(.+)$", re.M)
_RESOURCE_RE = re.compile(r"remaining=(-?[0-9.]+)")


def _mock_hypotheses(text):
    """Hypotheses drawn only from the digest's own rendered text.

    Never touches a lever menu -- a hypothesis is a read of the campaign's
    symptoms (its trend line, its unresolved signals), not a proposal of what
    to change; that step is `_mock_experiments`, which is the one that
    respects the lever blindfold.
    """
    hyps = []
    m = _TREND_RE.search(text)
    if m:
        trend, n_unresolved = m.group(1), m.group(2)
        hyps.append(
            "The digest's own trend read is: %s (%s unresolved signal(s) on "
            "the current round); the plan below tests whether that trend is "
            "recipe-driven or corpus-driven rather than assuming either."
            % (trend, n_unresolved))
    else:
        hyps.append(
            "No resolved trend line was found in this digest (too little "
            "history, or the state section was trimmed); the plan below "
            "proposes a controlled repeat of the current recipe as the only "
            "move that needs no assumption about the trend.")
    for name, sev, reason in _SIGNAL_LINE_RE.findall(text)[:3]:
        hyps.append(
            "Unresolved signal %s (%s): %s -- untested against a no-op "
            "control, so its effect on the target metric is not yet known."
            % (name, sev, reason.strip()))
    return hyps


def _mock_stop_rules(text):
    rules = [
        "stop the plan and escalate if any experiment trips a crit-severity "
        "signal before its seeds complete",
        "stop the plan if the domain's remaining SU envelope cannot cover "
        "the next queued experiment's est_su",
    ]
    m = _RESOURCE_RE.search(text)
    if m:
        rules.append(
            "resource check at plan time: the digest's resources section "
            "reported su_remaining=%s" % m.group(1))
    return rules


def _claim_seeds():
    """The campaign's own seed COUNT for a claim, from parallelism_rules.json.

    Reusing `parallelism.load_rules()` rather than a second literal `3` here
    is the anti-drift rule this package is built around: `seeds_for_claim` is
    already sealed data with its own reason in that file, and a planner that
    kept its own copy could drift from the parallelism layer that actually
    enforces it. Returns None when the rule is not declared -- never a
    guessed count.
    """
    rules = parallelism.load_rules()
    entry = (rules or {}).get("seeds_for_claim")
    if isinstance(entry, dict) and isinstance(entry.get("value"), int) and entry["value"] > 0:
        return entry["value"]
    return None


def _mock_experiments(lever_menu, opts):
    """(experiments, problems). See the module docstring for the blindfold:
    `lever_menu` is already None here whenever the digest did not show levers,
    so this function itself never has to re-derive that decision."""
    problems = []
    experiments = []
    if lever_menu:
        n_seeds = _claim_seeds()
        if n_seeds is None:
            problems.append(
                "parallelism_rules.json declares no seeds_for_claim; a lever "
                "experiment cannot be sized as a claim, so none were proposed "
                "from the visible lever menu")
        else:
            seeds = [101 + i for i in range(n_seeds)]
            max_n = int(opts.get("max_lever_experiments", 3))
            for lever in [l for l in lever_menu if isinstance(l, dict)][:max_n]:
                options = lever.get("options")
                variant = options[0] if isinstance(options, list) and options else options
                res = make_experiment(
                    recipe="lever:%s" % lever.get("id"),
                    params={"lever_id": lever.get("id"), "variant": variant},
                    seeds=seeds,
                    control=str(lever.get("control") or ""),
                    est_su=lever.get("est_su") if isinstance(lever.get("est_su"), (int, float))
                        and not isinstance(lever.get("est_su"), bool) else None,
                    risk=lever.get("risk"),
                    success_criterion=(
                        "the %d-seed mean differs from the control by more than "
                        "the recipe's sealed noise floor" % n_seeds),
                    stop_rule="stop this experiment if a seed's run does not complete")
                if res["ok"]:
                    experiments.append(res["experiment"])
                else:
                    problems.append(res["reason"])
    if not experiments:
        res = make_experiment(
            recipe="repeat_current_recipe",
            params={"note": "no lever table was visible to this plan"},
            seeds=[101, 102, 103],
            control="the current recipe, unchanged",
            est_su=None, risk="R2",
            success_criterion=(
                "the 3-seed mean differs from the control's most recently "
                "recorded mean by more than that recipe's sealed noise floor"),
            stop_rule="stop if any seed's run does not complete")
        if res["ok"]:
            experiments.append(res["experiment"])
        else:
            problems.append(res["reason"])
    return experiments, problems


def _base_plan(backend, digest_sha, version, hypotheses, experiments, stop_rules,
               simulated, problems=None):
    out = {
        "version": version,
        "created_from": {"digest_sha256": digest_sha},
        "hypotheses": hypotheses,
        "ordered_experiments": experiments,
        "stop_rules": stop_rules,
        "backend": backend,
        "simulated": bool(simulated),
        "refused": False,
        "reason": None,
    }
    if problems:
        out["problems"] = list(problems)
    return out


def _failed_plan(backend, digest_sha, reason):
    return {
        "version": None,
        "created_from": {"digest_sha256": digest_sha},
        "hypotheses": [], "ordered_experiments": [], "stop_rules": [],
        "backend": backend, "simulated": False,
        "refused": True, "reason": str(reason),
    }


# --- mock backend -------------------------------------------------------
def _mock_plan(digest, opts, backend_name="mock"):
    digest = digest if isinstance(digest, dict) else {}
    text = str(digest.get("text") or "")
    sections = digest.get("sections") if isinstance(digest.get("sections"), list) else []
    digest_sha = digest.get("sha256")
    digest_sha = str(digest_sha) if digest_sha is not None else None

    lever_menu = opts.get("lever_menu") if _levers_included(sections) else None
    lever_menu = lever_menu if isinstance(lever_menu, list) else None

    hyps = _mock_hypotheses(text)
    experiments, problems = _mock_experiments(lever_menu, opts)
    stop_rules = _mock_stop_rules(text)
    version = opts.get("version", 1)
    try:
        version = int(version)
    except (TypeError, ValueError):
        version = 1
    return _base_plan(backend_name, digest_sha, version, hyps, experiments,
                      stop_rules, True, problems)


# --- file backend: persistence on top of the same rule engine ---------------
_VERSION_RE = re.compile(r"^v(\d+)\.json$")


def _plans_dir(domain, root):
    return pathlib.Path(str(root)) / "results" / "framework" / "_brain" / str(domain) / "plans"


def _next_version_path(plans_dir):
    """(version, path) for a file that does not yet exist.

    Scans existing `vN.json` files for the current maximum rather than
    keeping a counter anywhere, so this is correct even the first time the
    directory is touched by a different process; the existence check in the
    loop is the actual guarantee against overwriting a version, not the scan.
    """
    plans_dir.mkdir(parents=True, exist_ok=True)
    versions = []
    for p in plans_dir.glob("v*.json"):
        m = _VERSION_RE.match(p.name)
        if m:
            versions.append(int(m.group(1)))
    version = (max(versions) + 1) if versions else 1
    path = plans_dir / ("v%d.json" % version)
    while path.exists():
        version += 1
        path = plans_dir / ("v%d.json" % version)
    return version, path


def _file_plan(digest, opts):
    domain = str(opts.get("domain") or "unknown-domain")
    root = opts.get("root") if opts.get("root") is not None else str(corpus.REPO)
    plans_dir = _plans_dir(domain, root)
    version, path = _next_version_path(plans_dir)
    inner_opts = dict(opts)
    inner_opts["version"] = version
    plan_ = _mock_plan(digest, inner_opts, backend_name="file")
    corpus.write_json(path, plan_)
    plan_ = dict(plan_)
    plan_["stored_path"] = str(path)
    return plan_


# --- cluster backend ---------------------------------------------------
def _cluster_prompt(digest_text):
    return (
        "You are the tier-2 planner for an unattended research campaign.\n"
        "Read the campaign digest below and reply with exactly one JSON "
        "object, no prose and no markdown fence, with these keys: "
        "\"hypotheses\" (a list of short strings), \"ordered_experiments\" "
        "(a list of objects, each with \"recipe\", \"params\", \"seeds\", "
        "\"control\", \"est_su\", \"risk\", \"success_criterion\", and "
        "\"stop_rule\"), and \"stop_rules\" (a list of strings). Every "
        "experiment must name a non-empty control and a non-empty "
        "success_criterion; an experiment missing either is dropped before "
        "it reaches the approval queue.\n\n"
        "=== CAMPAIGN DIGEST ===\n" + digest_text)


def _cluster_plan(digest, model, opts):
    digest = digest if isinstance(digest, dict) else {}
    digest_sha = digest.get("sha256")
    digest_sha = str(digest_sha) if digest_sha is not None else None
    backend_name = "cluster:%s" % model

    client = opts.get("client") or supervisor.OpenAICompatClient()
    num_ctx = opts.get("num_ctx") or supervisor.DEFAULT_NUM_CTX
    try:
        num_ctx = int(num_ctx)
    except (TypeError, ValueError):
        num_ctx = supervisor.DEFAULT_NUM_CTX

    prompt = _cluster_prompt(str(digest.get("text") or ""))
    try:
        res = client(prompt, model, num_ctx)
    except Exception as exc:                        # an injected/real client is untrusted input
        return _failed_plan(backend_name, digest_sha,
                            "model call raised %s: %s" % (type(exc).__name__, exc))

    if not isinstance(res, dict) or res.get("error"):
        reason = (res.get("error") if isinstance(res, dict) else None) or \
            "model call returned no usable result"
        return _failed_plan(backend_name, digest_sha, "model call failed: %s" % reason)

    obj = supervisor.json_from_text(res.get("text"))
    if not isinstance(obj, dict):
        return _failed_plan(backend_name, digest_sha,
                            "no JSON object found in the model's reply")

    hyps = [str(h) for h in (obj.get("hypotheses") or []) if str(h or "").strip()]
    stop_rules = [str(s) for s in (obj.get("stop_rules") or []) if str(s or "").strip()]
    experiments, problems = [], []
    raw_experiments = obj.get("ordered_experiments")
    for raw in (raw_experiments if isinstance(raw_experiments, list) else []):
        if not isinstance(raw, dict):
            problems.append("one ordered_experiments entry is not an object; dropped")
            continue
        out = make_experiment(
            recipe=raw.get("recipe"), params=raw.get("params"), seeds=raw.get("seeds"),
            control=raw.get("control"), est_su=raw.get("est_su"), risk=raw.get("risk"),
            success_criterion=raw.get("success_criterion"), stop_rule=raw.get("stop_rule"))
        if out["ok"]:
            experiments.append(out["experiment"])
        else:
            problems.append(out["reason"])

    version = opts.get("version", 1)
    try:
        version = int(version)
    except (TypeError, ValueError):
        version = 1
    return _base_plan(backend_name, digest_sha, version, hyps, experiments,
                      stop_rules, False, problems)


# --- the public entry point --------------------------------------------------
def plan(digest, backend, **opts):
    """One Plan for `digest`, computed by `backend` ("mock", "file", or
    "cluster:<model>"). Never raises: any internal defect comes back as a
    refused plan naming itself, the same contract `policy.authorize()` and
    `parallelism.plan()` keep for exactly the same reason -- a scheduler tick
    or a request handler calling this must not be brought down by a malformed
    digest, an unreachable endpoint, or a bug in this function itself."""
    try:
        return _plan(digest, backend, opts)
    except Exception as exc:
        digest_sha = None
        if isinstance(digest, dict):
            digest_sha = digest.get("sha256")
            digest_sha = str(digest_sha) if digest_sha is not None else None
        return _failed_plan(str(backend), digest_sha,
                            "plan() raised %s: %s" % (type(exc).__name__, exc))


def _plan(digest, backend, opts):
    backend = str(backend or "").strip()
    if not isinstance(digest, dict):
        return _failed_plan(backend, None, "digest is not an object")
    if backend == "mock":
        return _mock_plan(digest, opts)
    if backend == "file":
        return _file_plan(digest, opts)
    if backend.startswith("cluster:") and len(backend) > len("cluster:"):
        return _cluster_plan(digest, backend[len("cluster:"):], opts)
    digest_sha = digest.get("sha256")
    digest_sha = str(digest_sha) if digest_sha is not None else None
    return _failed_plan(backend, digest_sha,
                        "unknown backend %r; expected 'mock', 'file', or "
                        "'cluster:<model>'" % (backend,))


# --- structural validation ---------------------------------------------------
def validate(plan_obj):
    """(ok, problems) -- SHAPE only. This never judges whether a plan is any
    good: whether its hypotheses are correct, whether its experiments are the
    right ones to spend SU on, and whether its stop rules are sufficient are
    all outside what this function can see. A well-shaped bad plan and a
    malformed good idea are not this function's business to tell apart --
    only `experiments.verdict()` against a landed result, or a human, can."""
    if not isinstance(plan_obj, dict):
        return False, ["plan is not an object"]
    problems = []

    if plan_obj.get("version") is not None and not isinstance(plan_obj.get("version"), int):
        problems.append("version is %r, expected an int or null" % (plan_obj.get("version"),))

    cf = plan_obj.get("created_from")
    if not isinstance(cf, dict):
        problems.append("created_from is not an object")
    else:
        sha = cf.get("digest_sha256")
        if sha is not None and not isinstance(sha, str):
            problems.append("created_from.digest_sha256 is not a string or null")

    hyps = plan_obj.get("hypotheses")
    if not isinstance(hyps, list):
        problems.append("hypotheses is not a list")
    else:
        for i, h in enumerate(hyps):
            if not isinstance(h, str) or not h.strip():
                problems.append("hypotheses[%d] is not a non-empty string" % i)

    exps = plan_obj.get("ordered_experiments")
    if not isinstance(exps, list):
        problems.append("ordered_experiments is not a list")
    else:
        for i, e in enumerate(exps):
            if not isinstance(e, dict):
                problems.append("ordered_experiments[%d] is not an object" % i)
                continue
            if not str(e.get("recipe") or "").strip():
                problems.append("ordered_experiments[%d] names no recipe" % i)
            if not isinstance(e.get("params"), dict):
                problems.append("ordered_experiments[%d].params is not an object" % i)
            if not isinstance(e.get("seeds"), list):
                problems.append("ordered_experiments[%d].seeds is not a list" % i)
            if not str(e.get("control") or "").strip():
                problems.append("ordered_experiments[%d] names no control" % i)
            est = e.get("est_su")
            if est is not None and (isinstance(est, bool) or not isinstance(est, (int, float))):
                problems.append("ordered_experiments[%d].est_su is not a number or null" % i)
            risk = e.get("risk")
            if risk is not None and risk not in policy.RISKS:
                problems.append("ordered_experiments[%d].risk %r is not one of %s"
                                % (i, risk, policy.RISKS))
            if not str(e.get("success_criterion") or "").strip():
                problems.append("ordered_experiments[%d] names no success_criterion" % i)
            if not str(e.get("stop_rule") or "").strip():
                problems.append("ordered_experiments[%d] names no stop_rule" % i)

    stop_rules = plan_obj.get("stop_rules")
    if not isinstance(stop_rules, list):
        problems.append("stop_rules is not a list")
    else:
        for i, s in enumerate(stop_rules):
            if not isinstance(s, str) or not s.strip():
                problems.append("stop_rules[%d] is not a non-empty string" % i)

    if not isinstance(plan_obj.get("backend"), str) or not plan_obj.get("backend"):
        problems.append("backend is not a non-empty string")
    if not isinstance(plan_obj.get("simulated"), bool):
        problems.append("simulated is not a boolean")
    if not isinstance(plan_obj.get("refused"), bool):
        problems.append("refused is not a boolean")

    return (not problems), problems


# --- CLI ---------------------------------------------------------------------
def _main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="planner", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")

    pp = sub.add_parser("plan", help="build a plan from a digest JSON file")
    pp.add_argument("digest_file")
    pp.add_argument("--backend", default="mock")
    pp.add_argument("--domain", default=None)
    pp.add_argument("--root", default=None)
    pp.add_argument("--lever-menu-file", default=None)

    vp = sub.add_parser("validate", help="structural check of a plan JSON file")
    vp.add_argument("plan_file")

    args = ap.parse_args(argv)
    if args.cmd == "plan":
        digest = corpus.read_json(args.digest_file) or {}
        opts = {}
        if args.domain:
            opts["domain"] = args.domain
        if args.root:
            opts["root"] = args.root
        if args.lever_menu_file:
            opts["lever_menu"] = corpus.read_json(args.lever_menu_file) or []
        result = plan(digest, args.backend, **opts)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1 if result.get("refused") else 0
    if args.cmd == "validate":
        obj = corpus.read_json(args.plan_file)
        ok, problems = validate(obj)
        print(json.dumps({"ok": ok, "problems": problems}, indent=2))
        return 0 if ok else 1
    ap.print_help()
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
