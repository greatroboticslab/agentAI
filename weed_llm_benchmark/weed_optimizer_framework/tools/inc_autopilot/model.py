"""Shared data model of the INC autopilot (docs/INC_AUTOPILOT.md).

Every autopilot module exchanges these records as plain JSON-able dicts built
by the helpers below, so the lab ticker, the cluster verbs, the brain job and
the dashboard read the same shapes. Nothing here imports Ultralytics, torch
or the pinned inc modules.

Cite: the address of one value the autopilot read, so any claim can be
checked against the artifact:
    {"artifact": "<exp>/ledger.jsonl" | "<exp>/report.json" | "step1/select_summary.json" | ...,
     "line": <int> (ledger line, 1-based) or None,
     "pointer": "<JSON pointer into the artifact>" or None,
     "value": <the value read>}

Diagnosis: {"id": "D1", "name": "recipe_forgets", "fired": bool,
            "severity": "info" | "warn" | "crit", "summary": str,
            "cites": [Cite, ...], "levers": ["L1", ...], "exp": <exp or None>}

Proposal: {"id": <uuid hex>, "lever": "L1", "argv": [str, ...] (exact builder
           command), "params": {...}, "policy_action": "inc_build_pilot",
           "risk": "R0".."R4", "trigger": [diagnosis ids], "cites": [Cite, ...],
           "lit": [{"paper_id", "line", "quote"}], "control": str,
           "success": str, "falsifier": str, "est_gpu_hours": float,
           "proposed_by": "round-scheduler:inc-autopilot" | "tier2:<model>",
           "created_utc": str}
"""
from __future__ import annotations

import datetime
import os
import uuid
from pathlib import Path

# Lab-side state (the dashboard host). The results tree mirrors the brain
# package's convention: results/framework/_brain/<domain>/...
LAB_REPO = Path(os.environ.get("LAB_REPO", str(Path(__file__).resolve().parents[3])))
BRAIN_DIR = LAB_REPO / "results" / "framework" / "_brain"
DOMAIN = "weed"
CAMPAIGN_DIR = BRAIN_DIR / DOMAIN / "inc"
CAMPAIGN_LEDGER = CAMPAIGN_DIR / "inc_campaign.jsonl"        # append-only
SNAPSHOT_DIR = CAMPAIGN_DIR / "snapshots"                     # <exp>/<utc>.json
PLAN_DIR = CAMPAIGN_DIR / "plans"                             # brain plans pulled back
REPLAY_DIR = CAMPAIGN_DIR / "replay"                          # replay / prospective results

# Cluster-side (paths on /ocean; the lab never reads them directly, only via
# remote.py verbs over one batched ssh per tick).
CLUSTER_REPO = os.environ.get("CLUSTER_REPO", "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark")
CLUSTER_INC_DIR = CLUSTER_REPO + "/results/framework/inc"
CLUSTER_CAMPAIGN_DIR = CLUSTER_INC_DIR + "/_campaign"          # provenance/<exp>.json, plan inputs

AUTOPILOT_ACTOR = "round-scheduler:inc-autopilot"
DECISION_EXAM = "dev"          # the only exam any decision may read
REMOTE_MARK = "INCAP"          # remote.py prints exactly one line "INCAP <json>" per verb

# realloop build --increment-sources evidence reads no relevance file, so it never
# goes with --relevance: realloop refuses the pair, and so does every gate before
# it (levers.check_params, executor.resolve_params and render, validate, remote.py submit),
# each with this message.
EVIDENCE_WITH_RELEVANCE = ("--increment-sources evidence reads no relevance file: --relevance does not go with it "
                           "(realloop build refuses the two together)")

RISKS = ("R0", "R1", "R2", "R3", "R4")
SEVERITIES = ("info", "warn", "crit")


# --------------------------------------------------------------- test blindness
# The exams no decision may read come from the domain config (docs/FUNNEL_AUDIT.md
# 8.8 review; tools/funnel/domains/<domain>.json "exams"), not from a list typed
# here: a second domain's exam names would otherwise pass every guard silently.
# evidence.py, remote.py, brain_plan.py and validate.py build their lists from
# these two functions. `domain` is a domain name or a config path (default DOMAIN).

class DomainExamError(ValueError):
    """A domain config whose exam block cannot drive the test-blindness guard."""


def exam_splits(domain=None):
    """{"decision", "non_decision", "extra_non_decision"} of the domain config
    (funnel.domain.Domain.exam_splits). Refuses a config whose decision exam
    is not DECISION_EXAM: every autopilot decision reads that one exam."""
    from ..funnel import domain as FD
    sp = FD.load(domain or DOMAIN).exam_splits()
    if sp.get("decision") != DECISION_EXAM:
        raise DomainExamError("domain %s decides on %r, but the autopilot decides on %r only"
                              % (domain or DOMAIN, sp.get("decision"), DECISION_EXAM))
    return {"decision": sp["decision"], "non_decision": tuple(sp.get("non_decision") or ()),
            "extra_non_decision": tuple(sp.get("extra_non_decision") or ())}


def non_dev_exams(domain=None):
    """Every split no decision may read: the domain's non-decision exams, then
    its extra non-decision splits (the weed domain's H10d domain dev), each
    once, in config order. Fails closed: a config that names none is refused,
    because an empty list would let every value through."""
    sp = exam_splits(domain)
    names = list(sp["non_decision"])
    # The extra non-decision splits (the weed domain's H10d domain dev) are
    # blocked like exams (contract 8.7 review). Switching this off leaves the
    # other exams in place, so what catches it is the tests' own assertion
    # that the split is blocked, not an import-time refusal.
    if sp["extra_non_decision"]:  # funnel-mutation: M10
        names += list(sp["extra_non_decision"])
    out = []
    for exam in names:
        if exam != DECISION_EXAM and exam not in out:
            out.append(exam)
    if not out:
        raise DomainExamError("domain %s names no non-decision exam: the test-blindness guard would "
                              "block nothing" % (domain or DOMAIN))
    return tuple(out)


def domain_terms(domain=None):
    """The domain config's extra grep terms (funnel.domain.Domain.terms()
    substrings): validate.py's free-text leak scan adds '<term> test' for each,
    as it had 'cwd12 test' written into its pattern before."""
    from ..funnel import domain as FD
    return tuple((FD.load(domain or DOMAIN).raw or {}).get("domain_terms") or ())


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cite(artifact, value, line=None, pointer=None):
    if line is None and pointer is None:
        raise ValueError("a cite needs a ledger line or a JSON pointer")
    return {"artifact": str(artifact), "line": line, "pointer": pointer, "value": value}


def diagnosis(did, name, fired, severity, summary, cites, levers=(), exp=None):
    if severity not in SEVERITIES:
        raise ValueError("severity %r" % severity)
    # A diagnosis that fires without evidence is not a diagnosis.
    if fired and not cites:
        fired, severity = False, "info"
        summary = "unknown (no cited evidence): " + summary
    return {"id": did, "name": name, "fired": bool(fired), "severity": severity,
            "summary": summary, "cites": list(cites), "levers": list(levers), "exp": exp}


def proposal(lever, argv, params, policy_action, risk, trigger, cites, lit=(), control="",
             success="", falsifier="", est_gpu_hours=0.0, proposed_by=AUTOPILOT_ACTOR):
    if risk not in RISKS:
        raise ValueError("risk %r" % risk)
    return {"id": uuid.uuid4().hex, "lever": lever, "argv": [str(a) for a in argv],
            "params": dict(params), "policy_action": policy_action, "risk": risk,
            "trigger": list(trigger), "cites": list(cites), "lit": list(lit),
            "control": control, "success": success, "falsifier": falsifier,
            "est_gpu_hours": float(est_gpu_hours), "proposed_by": proposed_by,
            "created_utc": utc_now()}


def card(lever, title, trigger, cites, hypothesis="", why_menu_insufficient="", required_change="",
         cheapest_test="", control="", success_criterion="", lit=(), proposed_by=AUTOPILOT_ACTOR):
    """An R4 human research card (docs/INC_AUTOPILOT.md (b) X1-X9, (c)
    off_menu): never queued, never executed; shown to a person. Its fields
    are the brain output schema's off_menu fields, so a deterministic card
    and a brain card render the same. Added by levers.py (component 3)."""
    return {"id": uuid.uuid4().hex, "lever": lever, "risk": "R4", "title": title,
            "hypothesis": hypothesis, "why_menu_insufficient": why_menu_insufficient,
            "required_change": required_change, "cheapest_test": cheapest_test, "control": control,
            "success_criterion": success_criterion, "trigger": list(trigger), "cites": list(cites),
            "lit": list(lit), "proposed_by": proposed_by, "created_utc": utc_now()}
