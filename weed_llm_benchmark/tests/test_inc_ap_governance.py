#!/usr/bin/env python3
"""INC autopilot governance: the policy rows, the executor, the envelope rule, the budget.

docs/INC_AUTOPILOT.md, section d and step (g)6/8. The executor is the only path
that runs an autopilot action, so each rule it enforces is pinned here on a
temporary lab tree with a fake cluster hook (no ssh, no Mongo, no model):

- the twelve inc_* rows of brain/policy_actions.json: risk tiers, SU cost
  shape, bounds, and the authorize matrix per actor and risk;
- R0-R2 run directly for round-scheduler:inc-autopilot; R3 is filed with
  approvals.propose; R4 is never run, whoever asks;
- the owner's envelope rule: with autonomy 'envelope' and a complete replay
  pass on the current code, an R3 build of L1, L2, L5 or L8 is granted by the
  autopilot (lever, trigger diagnoses, cites, envelope balance) and runs
  authorised as the person who granted the envelope; off, or with no replay
  pass, it waits;
- what the envelope never grants: a brain's (or any other tier's) item, a
  non-INC action, a trigger that is not a fired diagnosis naming the lever,
  cites that are not the diagnosis's, a stop-loss in progress, parameters
  that are not the lever's, a resubmission that differs from the filed item,
  a build whose child_exp is not its own experiment;
- budget exhaustion (envelope, daily cap, committed estimates) blocks, and a
  run the execution log cannot record never happens; a crash mid-run stays
  charged;
- one approval id runs once, even with two callers racing; one direct
  proposal id runs once;
- the command that runs is the one the policy checked: parameters come out of
  the lever's argv, and the rendering must reproduce it;
- an automatic unblock needs a transient cause; cancel and sync-outer run
  through a person's approval;
- the replay gate: required cases, the governance files in the code hash, and
  the runner that records it;
- the SU-ledger writer reads nothing but exp, done, done_utc and gpu_hours;
- approvals.py keeps its old semantics (decide stays human-only, now under
  the log lock).

Run:  python3 tests/test_inc_ap_governance.py
"""
import copy
import json
import os
import pathlib
import re
import shlex
import shutil
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.brain import approvals as AP  # noqa: E402
from weed_optimizer_framework.tools.brain import policy as POL  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import budget as B  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import executor as X  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import model as M  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import remote as R  # noqa: E402

FAILURES = []
REPO = pathlib.Path(__file__).resolve().parents[1]
PILOT = REPO / "results" / "framework" / "inc" / "pilot_v1"
INC = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc/"
OWNER = "human:owner@example.org"
HUMAN = "human:harry@example.org"
AUTO = M.AUTOPILOT_ACTOR
BRAIN = "tier2:qwen3.8"
T0 = 1790000000.0
CITE = [M.cite("pilot_v1/report.json", 0.0, pointer="/chains/0/steps/1/decision/p_recipe")]
CITE_D4 = [M.cite("pilot_v1/report.json", 0.4286, pointer="/agreement/full/rate")]
CITE_D5 = [M.cite("pilot_v2/state.json", "transient", pointer="/blocked/chain:full/cause")]
CITE_D9 = [M.cite("pilot_v1/report.json", 6.67, pointer="/effective_warmup/max")]
CITE_X8 = [M.cite("step1/select_summary.json", 1540, pointer="/sizes/base_B")]
CITE_D14 = [M.cite("_loader", "test", pointer="/touched/0")]


def diag(did, levers, cites, exp="pilot_v1", **detail):
    return {"id": did, "name": "n_" + did, "fired": True, "severity": "warn", "summary": did,
            "cites": list(cites), "levers": list(levers), "exp": exp, "detail": detail}


# The fired diagnoses the ticker reports (the executor's `diagnoses` hook).
# No current rule names L8; DX8 is a synthetic diagnosis that does, so the L8
# path can be exercised.
FIRED = [diag("D1", ["L1", "X1"], CITE), diag("D4", ["L2"], CITE_D4),
         diag("D5", ["L7"], CITE_D5, exp="pilot_v2",
              units=[{"unit": "chain:full", "cause": "transient", "transient": True},
                     {"unit": "truth", "cause": "transient", "transient": True},
                     {"unit": "base", "cause": "failed_run", "transient": False}]),
         diag("D9", [], CITE_D9), diag("DX8", ["L8"], CITE_X8, exp=None)]
REPLAY_PASS = dict({c: "pass" for c in X.REPLAY_REQUIRED}, R2="skip", R4b="skip")
CAMP_OFF = {"name": "weedinc1", "autonomy": "off"}
CAMP_ENV = {"name": "weedinc1", "autonomy": "envelope", "autonomy_granted_by": OWNER}
INC_ACTIONS = ("inc_snapshot", "inc_report", "inc_advance", "inc_relevance_build",
               "inc_label_audit", "inc_unblock_transient", "inc_build_pilot",
               "inc_build_realloop", "inc_build_baseline", "inc_cancel_exp", "inc_sync_outer",
               "inc_lit_fetch")
BUILDS = ("inc_build_pilot", "inc_build_realloop", "inc_build_baseline")
SEG_RE = re.compile(r'echo "INCAP_SEG (\d+)"; python -u -m \S+ (.*?); echo "INCAP_SEG_END')


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


# --- a fake cluster and a temporary lab tree ---------------------------------------
class Fake(object):
    """slurm_sh: answers every remote.py verb line of a script with an INCAP record."""

    def __init__(self):
        self.calls, self.mode, self.delay, self.lock = [], "ok", 0.0, threading.Lock()

    def __call__(self, script, timeout=60):
        with self.lock:
            self.calls.append(script)
        if self.delay:
            time.sleep(self.delay)
        if self.mode == "raise":
            raise OSError("ssh: connect to host bridges2.psc.edu: timed out")
        if self.mode == "preamble":
            return {"ok": True, "stdout": "INCAP_PREAMBLE_FAILED\n", "stderr": "", "returncode": 0}
        out = ["Welcome to Bridges-2"]
        for m in SEG_RE.finditer(script):
            i, args = int(m.group(1)), shlex.split(m.group(2))
            out.append("INCAP_SEG %d" % i)
            if self.mode == "silent":
                out.append("INCAP_SEG_END %d 1" % i)
                continue
            rec = {"ok": self.mode != "refuse", "verb": args[0], "argv": args}
            if args[0] == "submit":
                rec["job_id"] = str(5000 + len(self.calls))
            if self.mode == "refuse":
                rec["error"] = "experiment is already built; an experiment is built once"
            out.append("a login profile line")
            out.append("INCAP " + json.dumps(rec))
            out.append("INCAP_SEG_END %d %d" % (i, 0 if rec["ok"] else 1))
        return {"ok": True, "stdout": "\n".join(out), "stderr": "", "returncode": 0}

    def verb_lines(self):
        return [shlex.split(m.group(2)) for s in self.calls for m in SEG_RE.finditer(s)]


class World(object):
    def __init__(self, **kw):
        self.dir = tempfile.mkdtemp(prefix="inc_ap_gov_")
        self.t = [T0]
        self.fake = Fake()
        self.fired = list(FIRED)
        kw.setdefault("resources", {"mongo_ok": True, "cluster_reachable": True})
        kw.setdefault("diagnoses", lambda name: self.fired)
        self.ctx = X.Context(slurm_sh=self.fake, lab_repo=self.dir, clock=lambda: self.t[0], **kw)

    def tick(self, s=1.0):
        self.t[0] += s

    def items(self):
        return AP.state("weed", root=self.dir)

    def replay(self, status="pass", code=None):
        cases = dict(REPLAY_PASS) if status == "pass" else {"R1": status}
        return X.record_replay_result(status, cases, ctx=self.ctx, code=code)

    def spend(self, exp, hours, campaign="weedinc1"):
        rep = {"exp": exp, "done": True, "done_utc": "2026-09-20T00:00:00Z",
               "gpu_hours": {"base": {"hours": hours, "runs": 3}}}
        return B.record_report_spend(rep, campaign, base_dir=self.ctx.su_base_dir)

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def sub(w, request, actor=AUTO, campaign=CAMP_OFF):
    w.tick()
    return X.submit(request, actor=actor, campaign=campaign, ctx=w.ctx)


# --- proposals as levers.py would write them ----------------------------------------
def l1(exp="pilot_v2", est=40.0, proposed_by=AUTO, **kw):
    argv = ["python", "-m", "weed_optimizer_framework.tools.inc.pilot", "build", "--exp", exp,
            "--replay-mode", "full"]
    p = M.proposal("L1", argv, {"exp": exp}, "inc_build_pilot", "R3", ["D1"], CITE,
                   est_gpu_hours=est, proposed_by=proposed_by)
    p.update(kw)
    return p


def l2(exp="real_v1", est=100.0, no_truth=False, lever="L2", cites=CITE_D4, **kw):
    argv = ["python", "-m", "weed_optimizer_framework.tools.inc.realloop", "build", "--exp", exp,
            "--base", INC + "step1/base_B.jsonl", "--replay-mode", "full", "--recipes", "full",
            "--relevance", INC + "step1/relevance.json"] + (["--no-truth"] if no_truth else [])
    params = {"exp": exp, "base": INC + "step1/base_B.jsonl", "replay_mode": "full",
              "recipes": "full", "relevance": INC + "step1/relevance.json"}
    p = M.proposal(lever, argv, params, "inc_build_realloop", "R3", ["D4"], cites,
                   est_gpu_hours=est)
    p.update(kw)
    return p


def l8(est=6.0, exp="base_b_v1", trigger=("DX8",), cites=CITE_X8):
    argv = ["python", "-m", "weed_optimizer_framework.tools.inc.pilot", "build-baseline", "--exp",
            exp, "--manifest", INC + "step1/base_B.jsonl"]
    return M.proposal("L8", argv, {"exp": exp, "manifest": INC + "step1/base_B.jsonl"},
                      "inc_build_baseline", "R3", list(trigger), cites, est_gpu_hours=est)


def l3(proposed_by=AUTO):
    return M.proposal("L3", ["sbatch", "run_inc_relevance.sh", "build"], {"est_gpu_hours": 1.0},
                      "inc_relevance_build", "R2", ["D2"], CITE, est_gpu_hours=1.0,
                      proposed_by=proposed_by)


def l7(exp="pilot_v2", unit="chain:full", cause="transient"):
    argv = ["python", "-m", "weed_optimizer_framework.tools.inc.driver", "unblock", "--exp", exp,
            "--unit", unit, "--reason", "auto: " + cause]
    return M.proposal("L7", argv, {"exp": exp, "unit": unit, "cause": cause},
                      "inc_unblock_transient", "R2", ["D5"], CITE_D5)


def plain(action, **params):
    return {"policy_action": action, "params": params}


def audit_command():
    """run_inc_audit.sh's own documented pilot command (lines 30-36), variables expanded."""
    lines = [ln[1:].strip() for ln in (REPO / "run_inc_audit.sh").read_text().splitlines()
             if ln.startswith("#")]
    repo = next(ln.split("=", 1)[1] for ln in lines if ln.startswith("REPO="))
    start = next(i for i, ln in enumerate(lines)
                 if ln.startswith("sbatch run_inc_audit.sh --trusted"))
    cmd = ""
    for ln in lines[start:]:
        cmd += " " + ln.rstrip("\\").strip()
        if not ln.endswith("\\"):
            break
    inc = repo + "/results/framework/inc"
    cmd = cmd.replace("$M", inc + "/pilot_v1/manifests").replace("$INC", inc)
    return shlex.split(cmd)[1:]                     # without 'sbatch'


# ===================================================================================
def section_policy_rows():
    print("policy rows (step (g)6)")
    rows = {a: POL.describe(a) for a in INC_ACTIONS}
    check("all twelve inc rows load", all(r["known"] for r in rows.values()),
          [a for a, r in rows.items() if not r["known"]])
    check("the whole table loads with no errors", POL.errors() == [], POL.errors())
    want = {"inc_snapshot": "R0", "inc_report": "R0", "inc_lit_fetch": "R0", "inc_advance": "R1",
            "inc_relevance_build": "R2", "inc_label_audit": "R2", "inc_unblock_transient": "R2",
            "inc_build_pilot": "R3", "inc_build_realloop": "R3", "inc_build_baseline": "R3",
            "inc_cancel_exp": "R3", "inc_sync_outer": "R3"}
    check("risk tiers are the contract's (section d)",
          {a: rows[a]["risk"] for a in want} == want, {a: rows[a]["risk"] for a in want})
    for a in BUILDS:
        f = rows[a]["est_su"]
        check("%s is priced as est_gpu_hours V100 GPU-hours with no default" % a,
              f.get("hours_param") == "est_gpu_hours" and f.get("gpu_count") == 1
              and "v100" in f.get("gpu_type", "") and "hours_default" not in f, f)
        check("%s bounds est_gpu_hours to [0, 200]" % a,
              rows[a]["param_bounds"]["est_gpu_hours"] == {"type": "float", "min": 0.0,
                                                           "max": 200.0})
        check("round-scheduler may file %s; tier0/tier1 may not ask" % a,
              "round-scheduler" in rows[a]["allowed_tiers"]
              and not {"tier0", "tier1"} & set(rows[a]["allowed_tiers"]))
    check("a build costs its estimate at 1.0 SU per V100 GPU-hour",
          POL.estimate_su("inc_build_pilot", {"est_gpu_hours": 40.0})["su"] == 40.0)
    check("a build with no estimate has an unknown cost, not 0",
          POL.estimate_su("inc_build_pilot", {})["su"] is None)
    check("relevance and audit cost their scripts' walltime (1 h, 2 h)",
          POL.estimate_su("inc_relevance_build", {})["su"] == 1.0
          and POL.estimate_su("inc_label_audit", {})["su"] == 2.0)
    check("advance, unblock, snapshot and report are free",
          all(POL.estimate_su(a, {})["su"] == 0.0 for a in
              ("inc_advance", "inc_unblock_transient", "inc_snapshot", "inc_report")))

    def ok(action, p):
        return POL.authorize(HUMAN, action, p)["allowed"]

    good = {"exp": "pilot_v2", "replay_mode": "full", "est_gpu_hours": 40.0}
    check("a well-formed pilot build is allowed", ok("inc_build_pilot", good))
    for est in (200.5, -1.0, "40", True):
        check("est_gpu_hours %r is refused" % (est,),
              not ok("inc_build_pilot", dict(good, est_gpu_hours=est)))
    for bad in ("../x", "a b", "-x", "x" * 65, "pilot;rm", ""):
        check("experiment name %r is refused" % bad, not ok("inc_build_pilot", dict(good, exp=bad)))
    check("replay mode 'fast' is refused",
          not ok("inc_build_pilot", dict(good, replay_mode="fast")))
    rl = {"exp": "real_v1", "replay_mode": "full", "recipes": "full,lora", "est_gpu_hours": 90.0,
          "relevance": INC + "step1/relevance.json", "base": INC + "step1/base_B.jsonl"}
    check("a realloop build with base and relevance under INC_DIR is allowed",
          ok("inc_build_realloop", rl))
    check("recipes out of the table's order are refused",
          not ok("inc_build_realloop", dict(rl, recipes="lora,full")))
    check("a repeated recipe is refused", not ok("inc_build_realloop", dict(rl, recipes="full,full")))
    check("a relevance path with '..' is refused",
          not ok("inc_build_realloop", dict(rl, relevance=INC + "step1/../../x/relevance.json")))
    check("a relevance path outside INC_DIR is refused",
          not ok("inc_build_realloop", dict(rl, relevance="/tmp/relevance.json")))
    check("size above the bound is refused", not ok("inc_build_realloop", dict(rl, size=50001)))
    check("no_truth is 0 or 1", not ok("inc_build_realloop", dict(rl, no_truth=2)))
    au = {"trusted": INC + "pilot_v1/manifests/P0.jsonl",
          "audit": "I1=%spilot_v1/manifests/I1.jsonl,Bswap=%spilot_v1/manifests/Bswap.jsonl"
                   % (INC, INC),
          "out": INC + "pilot_v1/audit/label_audit.json"}
    check("a label audit inside INC_DIR is allowed", ok("inc_label_audit", au))
    check("an audit pair with a shell metacharacter is refused",
          not ok("inc_label_audit", dict(au, audit=au["audit"] + ";rm")))
    check("an audit pair outside INC_DIR is refused",
          not ok("inc_label_audit", dict(au, audit="I1=/etc/passwd.jsonl")))
    check("an audit output outside <exp>/audit/ is refused",
          not ok("inc_label_audit", dict(au, out=INC + "pilot_v1/report.json")))
    check("an unblock of an unknown unit is refused",
          not ok("inc_unblock_transient", {"exp": "p", "unit": "chain:sgd", "cause": "transient"}))
    check("an unblock cause with other characters is refused",
          not ok("inc_unblock_transient", {"exp": "p", "unit": "truth", "cause": "Transient!"}))
    check("--force is not a relevance parameter", not ok("inc_relevance_build", {"force": 1}))
    check("a lit fetch takes an arXiv id", ok("inc_lit_fetch", {"arxiv_id": "2403.08763",
                                                               "paper_id": "ibrahim2024"}))
    check("a lit fetch of a path is refused",
          not ok("inc_lit_fetch", {"arxiv_id": "../etc", "paper_id": "x"}))


def section_authorize_matrix():
    print("authorize matrix per actor and risk")
    valid = {
        "inc_snapshot": {"exp": "pilot_v1"}, "inc_report": {"exp": "pilot_v1"},
        "inc_lit_fetch": {"arxiv_id": "2403.08763", "paper_id": "ibrahim2024"},
        "inc_advance": {"exp": "pilot_v1"},
        "inc_relevance_build": {}, "inc_label_audit": {
            "trusted": INC + "pilot_v1/manifests/P0.jsonl",
            "audit": "I1=" + INC + "pilot_v1/manifests/I1.jsonl",
            "out": INC + "pilot_v1/audit/label_audit.json"},
        "inc_unblock_transient": {"exp": "pilot_v1", "unit": "truth", "cause": "transient"},
        "inc_build_pilot": {"exp": "pilot_v2", "replay_mode": "full", "est_gpu_hours": 40.0},
        "inc_build_realloop": {"exp": "real_v1", "replay_mode": "full", "recipes": "full",
                               "est_gpu_hours": 90.0},
        "inc_build_baseline": {"exp": "base_b_v1", "manifest": INC + "step1/base_B.jsonl",
                               "est_gpu_hours": 6.0},
        "inc_cancel_exp": {"exp": "pilot_v2"}, "inc_sync_outer": {}}
    actors = (AUTO, BRAIN, "tier1:glm-4.7-flash", "tier0:gemma4", HUMAN)
    D, P, N = "direct", "propose", "refuse"
    expect = {
        "inc_snapshot": (D, D, D, D, D), "inc_report": (D, D, D, D, D),
        "inc_lit_fetch": (D, N, N, N, D), "inc_advance": (D, N, N, N, D),
        "inc_relevance_build": (D, P, N, N, D), "inc_label_audit": (D, P, N, N, D),
        "inc_unblock_transient": (D, N, N, N, D),
        "inc_build_pilot": (N, P, N, N, D), "inc_build_realloop": (N, P, N, N, D),
        "inc_build_baseline": (N, P, N, N, D),
        "inc_cancel_exp": (N, N, N, N, D), "inc_sync_outer": (N, N, N, N, D)}
    bad = []
    for action, row in expect.items():
        for actor, want in zip(actors, row):
            d = POL.authorize(actor, action, valid[action])
            got = N if not d["allowed"] else (P if d["needs_approval"] else D)
            if got != want:
                bad.append((action, actor, want, got, d["reasons"]))
    check("every (actor, action) cell is the expected direct / propose / refuse", not bad, bad)
    d = POL.authorize(AUTO, "inc_build_pilot", valid["inc_build_pilot"])
    check("the autopilot's R3 refusal names its ceiling, not the row",
          "no authority at R3" in " ".join(d["reasons"]), d["reasons"])

    # The executor's outcome per actor (autonomy off).
    cases = [
        (AUTO, plain("inc_snapshot", exp="pilot_v1"), "executed"),
        (AUTO, plain("inc_advance", exp="pilot_v1"), "executed"),
        (AUTO, l3(), "executed"),
        (AUTO, l7(), "executed"),
        (AUTO, l1(), "filed"),
        (AUTO, l8(), "filed"),
        (BRAIN, plain("inc_snapshot", exp="pilot_v1"), "executed"),
        (BRAIN, l3(proposed_by=BRAIN), "filed"),
        (BRAIN, l1(proposed_by=BRAIN), "filed"),
        (BRAIN, plain("inc_advance", exp="pilot_v1"), "refused"),
        ("tier0:gemma4", plain("inc_relevance_build"), "refused"),
        ("tier1:glm-4.7-flash", plain("inc_snapshot", exp="pilot_v1"), "executed"),
        (HUMAN, l1(), "executed"),
        (HUMAN, plain("inc_relevance_build"), "executed"),
        (HUMAN, plain("inc_cancel_exp", exp="pilot_v2"), "executed"),
        (HUMAN, plain("inc_sync_outer"), "executed"),
        (AUTO, plain("inc_cancel_exp", exp="pilot_v2"), "filed"),
        (AUTO, plain("inc_sync_outer"), "filed"),
    ]
    for actor, req, want in cases:
        w = World()
        try:
            r = sub(w, req, actor=actor)
            check("executor: %s -> %s is %s" % (actor, req.get("policy_action"), want),
                  r["status"] == want, (r["status"], r["reasons"]))
            if want == "filed":
                item = w.items().get(r["approval_id"]) or {}
                check("  the filed item is pending with its requester and risk",
                      item.get("status") == "pending" and item.get("requested_by") == actor
                      and item.get("risk") == POL.risk_of(req["policy_action"])
                      and not w.fake.calls, item)
        finally:
            w.close()


def section_envelope():
    print("envelope autonomy on / off")
    w = World()
    try:
        r = sub(w, l1(), campaign=CAMP_OFF)
        item = w.items()[r["approval_id"]]
        check("autonomy off: an L1 build is filed and waits", r["status"] == "filed"
              and item["status"] == "pending" and not w.fake.calls, r["reasons"])
        check("  the reason says autonomy is off",
              any("autonomy is off" in x for x in r["reasons"]), r["reasons"])
        check("  the item carries lever, trigger, cites and argv",
              item["context"]["lever"] == "L1" and item["context"]["trigger"] == ["D1"]
              and item["context"]["cites"] == CITE and "--replay-mode" in item["context"]["argv"])
    finally:
        w.close()

    w = World()
    try:
        w.replay()
        r = sub(w, l1(), campaign=CAMP_ENV)
        check("autonomy on, replay passed: an L1 build runs at once", r["status"] == "executed",
              r["reasons"])
        item = w.items()[r["approval_id"]]
        g = item.get("grant") or {}
        check("  the item is approved by the autopilot on the envelope basis",
              item["status"] == "approved" and item["decided_by"] == AUTO
              and item.get("decision_basis") == "envelope", item)
        check("  the grant records lever, trigger, cites and the person whose envelope it is",
              g.get("lever") == "L1" and g.get("trigger") == ["D1"] and g.get("cites") == CITE
              and g.get("authority") == OWNER)
        env = g.get("envelope") or {}
        check("  the grant records the envelope balance",
              env.get("envelope_su") == 300.0 and env.get("remaining_su") == 300.0
              and env.get("est_su") == 40.0 and env.get("balance_after_su") == 260.0, env)
        check("  it ran authorised as the person who granted the envelope",
              r["authorized_as"] == OWNER and r["basis"] == "envelope")
        check("  the execution is recorded done with its job id",
              (item.get("execution") or {}).get("phase") == "done"
              and item["execution"]["outcome"]["job_ids"] == r["job_ids"] and r["job_ids"])
        line = w.fake.verb_lines()[-1]
        check("  remote.py submit carries the approval, the decider and the trigger",
              line[:2] == ["submit", "build"] and ["--approval-id", r["approval_id"]] ==
              line[line.index("--approval-id"):line.index("--approval-id") + 2]
              and "--decided-by" in line and line[line.index("--decided-by") + 1] == AUTO
              and line[line.index("--trigger") + 1] == "D1", line)
        check("  and the builder arguments the lever card shows",
              line[line.index("--") + 1:] == ["pilot", "build", "--exp", "pilot_v2",
                                              "--replay-mode", "full"], line)
        check("  one ssh call", len(w.fake.calls) == 1)
        g = w.items()[r["approval_id"]]["grant"]
        check("  the grant names the fired diagnosis it checked and what it verified",
              [d["id"] for d in g["diagnoses"]] == ["D1"] and g["diagnoses"][0]["via"] == "levers"
              and g["checks"]["trigger_diagnoses_fired"] is True
              and g["checks"]["cites_from_diagnoses"] == 1
              and g["checks"]["cites_not_verified_here"] == 0, g.get("checks"))
        r2 = sub(w, l8(), campaign=CAMP_ENV)
        check("an L8 baseline runs under the envelope when a fired diagnosis names L8",
              r2["status"] == "executed", r2["reasons"])
        r3 = sub(w, l2(est=60.0), campaign=CAMP_ENV)
        check("an L2 real loop runs when it fits the daily cap", r3["status"] == "executed",
              r3["reasons"])
        r4 = sub(w, l2(exp="real_v2", est=10.0, no_truth=True), campaign=CAMP_ENV)
        check("--no-truth (L6) is not covered: filed",
              r4["status"] == "filed" and any("L6" in x for x in r4["reasons"]), r4["reasons"])
        r5 = sub(w, l2(exp="real_v3", est=10.0, cites=[]), campaign=CAMP_ENV)
        check("a proposal with no cited diagnosis is filed",
              r5["status"] == "filed" and any("cited diagnosis" in x for x in r5["reasons"]))
        r6 = sub(w, l1(exp="pilot_v4", est=5.0, lever="L2"), campaign=CAMP_ENV)
        check("a lever that does not match its action is filed",
              r6["status"] == "filed" and any("not covered" in x for x in r6["reasons"]))
        r7 = sub(w, l1(exp="pilot_v5", est=5.0, proposed_by=BRAIN), actor=BRAIN, campaign=CAMP_ENV)
        check("a brain (tier2) proposal is never self-approved",
              r7["status"] == "filed" and w.items()[r7["approval_id"]]["status"] == "pending")
        r8 = sub(w, l1(exp="pilot_v6", est=0.0), campaign=CAMP_ENV)
        check("a build with no positive estimate is filed", r8["status"] == "filed",
              r8["reasons"])
        for camp, what in ((dict(CAMP_ENV, autonomy_granted_by=None), "no granting person"),
                           (dict(CAMP_ENV, autonomy_granted_by="tier1:x"), "a model as grantor"),
                           (dict(CAMP_ENV, autonomy_granted_by="human"), "a bare 'human'")):
            r9 = sub(w, l1(exp="pilot_v7", est=5.0, id=None), campaign=camp)
            check("autonomy with %s is filed, for that reason" % what, r9["status"] == "filed"
                  and any("no person who granted the envelope" in x for x in r9["reasons"]),
                  r9["reasons"])
        r10 = sub(w, l1(exp="pilot_v8", est=5.0), campaign=dict(CAMP_ENV, paused_reason="D7"))
        check("a paused campaign refuses the autopilot's R1+ requests",
              r10["status"] == "refused" and "paused" in r10["reasons"][0])
        r11 = sub(w, plain("inc_snapshot", exp="pilot_v1"), campaign=dict(CAMP_ENV,
                                                                         paused_reason="D7"))
        check("  but still reads (R0)", r11["status"] == "executed")
    finally:
        w.close()

    w = World()
    try:
        w.replay()
        got = [sub(w, l1(exp="lc_%d" % i, est=10.0), campaign=CAMP_ENV) for i in range(4)]
        check("stop-loss: a 4th submission of the same lever is not self-approved",
              [r["status"] for r in got] == ["executed", "executed", "executed", "filed"]
              and any("already ran 3 times" in x for x in got[3]["reasons"]),
              [(r["status"], r["reasons"]) for r in got])
    finally:
        w.close()

    w = World()
    try:
        item = AP.propose("weed", "inc_build_pilot", {"exp": "p9", "replay_mode": "full",
                                                      "est_gpu_hours": 5.0}, "R3", AUTO,
                          "why", T0, root=w.dir, est_su=5.0,
                          context={"lever": "L1", "trigger": ["D1"], "cites": CITE,
                                   "proposed_by": AUTO})["item"]
        grant = {"lever": "L1", "trigger": ["D1"], "cites": CITE, "authority": OWNER,
                 "envelope": {"remaining_su": 300.0, "est_su": 5.0}}
        check("a grant whose estimate is not the item's is refused",
              not AP.approve_within_envelope(
                  "weed", item["id"], AUTO,
                  dict(grant, envelope={"remaining_su": 300.0, "est_su": 1.0}), "r", T0 + 1,
                  root=w.dir)["ok"])
        check("a grant whose trigger is not the item's is refused",
              not AP.approve_within_envelope("weed", item["id"], AUTO, dict(grant, trigger=["D14"]),
                                             "r", T0 + 1, root=w.dir)["ok"])
        bare = AP.propose("weed", "inc_build_pilot", {"exp": "p8"}, "R3", AUTO, "why", T0,
                          root=w.dir, est_su=5.0)["item"]
        check("a grant on an item filed with no lever context is refused",
              not AP.approve_within_envelope("weed", bare["id"], AUTO, grant, "r", T0 + 1,
                                             root=w.dir)["ok"])
        bi = AP.propose("weed", "inc_build_pilot", {"exp": "p7"}, "R3", BRAIN, "why", T0,
                        root=w.dir, est_su=5.0,
                        context={"lever": "L1", "trigger": ["D1"], "cites": CITE})["item"]
        r = AP.approve_within_envelope("weed", bi["id"], AUTO, grant, "r", T0 + 1, root=w.dir)
        check("a grant on an item a brain requested is refused",
              not r["ok"] and "waits for a person" in r["reason"], r)
        adopted = AP.propose("weed", "inc_build_pilot", {"exp": "p6"}, "R3", AUTO, "why", T0,
                             root=w.dir, est_su=5.0,
                             context={"lever": "L1", "trigger": ["D1"], "cites": CITE,
                                      "proposed_by": BRAIN})["item"]
        r = AP.approve_within_envelope("weed", adopted["id"], AUTO, grant, "r", T0 + 1, root=w.dir)
        check("a grant on a brain's proposal the autopilot re-filed is refused",
              not r["ok"] and "never self-granted" in r["reason"], r)
        other = AP.propose("weed", "roboflow_generate_versions", {"n": 1}, "R3", AUTO, "why", T0,
                           root=w.dir, est_su=5.0,
                           context={"lever": "L1", "trigger": ["D1"], "cites": CITE})["item"]
        r = AP.approve_within_envelope("weed", other["id"], AUTO, grant, "r", T0 + 1, root=w.dir)
        check("a grant of an action outside the INC builds is refused",
              not r["ok"] and "decided by a person" in r["reason"], r)
        check("decide() still refuses the autopilot",
              not AP.decide("weed", item["id"], "approve", AUTO, "fine", T0 + 1,
                            root=w.dir)["ok"])
        check("a person cannot record an envelope grant",
              not AP.approve_within_envelope("weed", item["id"], HUMAN, grant, "r", T0 + 1,
                                             root=w.dir)["ok"])
        check("a model cannot record an envelope grant",
              not AP.approve_within_envelope("weed", item["id"], "tier1:x", grant, "r", T0 + 1,
                                             root=w.dir)["ok"])
        check("a grant without cites is refused",
              not AP.approve_within_envelope("weed", item["id"], AUTO, dict(grant, cites=[]),
                                             "r", T0 + 1, root=w.dir)["ok"])
        check("a grant over the balance is refused",
              not AP.approve_within_envelope(
                  "weed", item["id"], AUTO,
                  dict(grant, envelope={"remaining_su": 4.0, "est_su": 5.0}), "r", T0 + 1,
                  root=w.dir)["ok"])
        r2 = AP.propose("weed", "inc_relevance_build", {}, "R2", AUTO, "why", T0 + 2,
                        root=w.dir)["item"]
        check("a grant covers R3 items only",
              not AP.approve_within_envelope("weed", r2["id"], AUTO, grant, "r", T0 + 3,
                                             root=w.dir)["ok"])
        ok = AP.approve_within_envelope("weed", item["id"], AUTO, grant, "r", T0 + 4, root=w.dir)
        check("the autopilot's grant on a pending R3 item is recorded", ok["ok"]
              and ok["item"]["status"] == "approved", ok)
        check("a second grant on the same item is refused",
              not AP.approve_within_envelope("weed", item["id"], AUTO, grant, "r", T0 + 5,
                                             root=w.dir)["ok"])
    finally:
        w.close()


def section_envelope_scope():
    print("what the envelope grants, and what it leaves to a person")
    w = World()
    try:
        w.replay()
        p = l1(exp="pilot_b1", est=10.0, proposed_by=BRAIN)
        r1 = sub(w, p, actor=BRAIN, campaign=CAMP_ENV)
        r2 = sub(w, dict(p, proposed_by=None), actor=AUTO, campaign=CAMP_ENV)
        item = w.items()[r1["approval_id"]]
        check("a brain's filed item resubmitted by the autopilot is not granted",
              r1["status"] == "filed" and r2["status"] == "filed"
              and r2["approval_id"] == r1["approval_id"] and item["status"] == "pending"
              and not w.fake.calls
              and any("requested by 'tier2:qwen3.8'" in x for x in r2["reasons"]), r2["reasons"])
        AP.decide("weed", item["id"], "approve", HUMAN, "reviewed", T0 + 50, root=w.dir)
        r3 = X.run_approved(CAMP_ENV, w.ctx)
        check("  once a person approves it, it runs authorised as that person",
              [x["status"] for x in r3] == ["executed"] and r3[0]["authorized_as"] == HUMAN
              and r3[0]["basis"] == "approval", [(x["status"], x["reasons"]) for x in r3])
        forged = AP.propose("weed", "inc_build_pilot", {"exp": "pilot_b2", "replay_mode": "full",
                                                        "est_gpu_hours": 5.0}, "R3", BRAIN, "why",
                            T0 + 60, root=w.dir, est_su=5.0,
                            context={"campaign": "weedinc1", "lever": "L1", "trigger": ["D1"],
                                     "cites": CITE, "proposal_id": "f1"})["item"]
        with open(AP.path("weed", root=w.dir), "a") as fh:
            fh.write(json.dumps({"kind": "grant", "id": forged["id"], "decided_by": AUTO,
                                 "decision": "approved", "basis": "envelope", "ts": T0 + 61,
                                 "grant": {}}) + "\n")
        check("a hand-written grant line on a brain's item is not applied by the fold",
              w.items()[forged["id"]]["status"] == "pending")
    finally:
        w.close()

    def filed_for(label, req, needle, world=None, camp=CAMP_ENV):
        ww = world or World()
        try:
            if world is None:
                ww.replay()
            before = len(ww.fake.calls)
            r = sub(ww, req, campaign=camp)
            check(label, r["status"] == "filed" and len(ww.fake.calls) == before
                  and any(needle in x for x in r["reasons"]), (r["status"], r["reasons"]))
            return r
        finally:
            if world is None:
                ww.close()

    junk = [M.cite("nowhere.json", 42, pointer="/made/up")]
    filed_for("a stop-loss diagnosis as trigger, with a made-up cite, is not granted",
              l8(exp="base_b_v9", trigger=("D14",), cites=junk), "trigger D14")
    filed_for("a trigger that names another lever is not granted (D4 names L2, not L8)",
              l8(exp="base_b_v9", trigger=("D4",), cites=CITE_D4), "trigger D4")
    filed_for("a made-up cite in place of the diagnosis's own is not granted",
              l1(exp="pilot_j1", est=5.0, cites=junk), "every cite carried")
    filed_for("a supporting diagnosis alone does not trigger a build (D9 supports L1)",
              l1(exp="pilot_j2", est=5.0, trigger=["D9"], cites=CITE_D9), "names lever L1 itself")
    r = filed_for("a malformed cite is not granted",
                  l1(exp="pilot_j3", est=5.0, cites=CITE + [{"artifact": "x", "value": 1}]),
                  "cite shape")
    w = World()
    try:
        w.replay()
        r = sub(w, l1(exp="pilot_j4", est=5.0, trigger=["D1", "D9"], cites=CITE + CITE_D9),
                campaign=CAMP_ENV)
        check("a trigger that names the lever plus one that supports it is granted",
              r["status"] == "executed" and [d["via"] for d in
                                             w.items()[r["approval_id"]]["grant"]["diagnoses"]]
              == ["levers", "supports"], r["reasons"])
        w.fired = FIRED + [diag("D14", ["OP_HALT"], CITE_D14)]
        filed_for("while a stop-loss diagnosis is firing nothing is granted",
                  l1(exp="pilot_j5", est=5.0), "stop-loss diagnosis D14", world=w)
        w.fired = FIRED + [diag("D5", ["OP_PAUSE"], CITE_D5, exp="pilot_v3")]
        filed_for("  nor while any diagnosis asks to pause", l1(exp="pilot_j6", est=5.0),
                  "stop-loss diagnosis D5", world=w)
        w.fired = [d for d in FIRED if d["id"] != "D1"]
        filed_for("a trigger that is not firing now is not granted", l1(exp="pilot_j7", est=5.0),
                  "trigger D1", world=w)
    finally:
        w.close()
    w = World(diagnoses=None)
    try:
        w.replay()
        filed_for("with no fired-diagnosis record from the ticker nothing is granted",
                  l1(exp="pilot_j8", est=5.0), "no fired-diagnosis record", world=w)
    finally:
        w.close()

    bad_l1 = l1(exp="pilot_k1", est=5.0,
                argv=["python", "-m", "weed_optimizer_framework.tools.inc.pilot", "build", "--exp",
                      "pilot_k1", "--replay-mode", "sample"])
    bad_l1["params"] = {"exp": "pilot_k1"}
    filed_for("an 'L1' whose argv is not L1 (replay sample) is not granted", bad_l1,
              "lever L1 fixes replay_mode")
    l5 = l2(exp="real_k2", est=5.0, lever="L5")
    filed_for("an 'L5' with no --size (L5 requires it) is not granted", l5, "requires size")

    w = World()
    try:
        w.replay()
        camp = dict(CAMP_ENV, daily_cap_su=100.0)
        p = l1(exp="pilot_d1", est=150.0)
        r1 = sub(w, p, campaign=camp)
        r2 = sub(w, l1(exp="pilot_d1", est=5.0, id=p["id"]), campaign=camp)
        item = w.items()[r1["approval_id"]]
        check("a proposal id resubmitted with another price is refused, and nothing is granted",
              r1["status"] == "filed" and r2["status"] == "refused"
              and "a changed proposal is a new proposal" in r2["reasons"][0]
              and item["status"] == "pending" and "grant" not in item, r2["reasons"])
        r3 = sub(w, l1(exp="pilot_d1", est=150.0, id=p["id"], cites=CITE + CITE_D9), campaign=camp)
        check("  likewise with other cites", r3["status"] == "refused"
              and "cites" in r3["reasons"][0], r3["reasons"])
        r4 = sub(w, l1(exp="pilot_c1", est=150.0, child_exp="pilot_v1"), campaign=camp)
        check("a build whose child_exp is not its own --exp is refused",
              r4["status"] == "refused" and "child_exp" in r4["reasons"][0], r4["reasons"])
        r5 = sub(w, l1(exp="pilot_c2", est=5.0, child_exp="pilot_c2"), campaign=camp)
        check("  and one that names its own experiment runs", r5["status"] == "executed",
              r5["reasons"])
    finally:
        w.close()


def section_log_integrity():
    print("nothing runs that the execution log cannot record and charge")
    w = World()
    try:
        w.replay()
        camp = dict(CAMP_ENV, envelope_su=50.0)
        w.ctx.exec_log.parent.mkdir(parents=True, exist_ok=True)
        w.ctx.exec_log.mkdir()                        # a directory: every append fails
        got = [sub(w, l1(exp="ov_%d" % i, est=40.0), campaign=camp) for i in range(3)]
        check("an unwritable execution log: no envelope build is granted or run",
              [r["status"] for r in got] == ["filed"] * 3 and not w.fake.calls
              and all(w.items()[r["approval_id"]]["status"] == "pending" for r in got),
              [(r["status"], r["reasons"]) for r in got])
        r = sub(w, l3(), campaign=camp)
        check("  nor a direct R2 action", r["status"] == "refused" and not w.fake.calls
              and "nothing ran" in r["reasons"][0], r["reasons"])
        AP.decide("weed", got[0]["approval_id"], "approve", HUMAN, "go", T0 + 50, root=w.dir)
        r = X.execute_approved(got[0]["approval_id"], camp, w.ctx)
        check("  nor a person's approval, which stays unexecuted for later",
              r["status"] == "refused" and not w.fake.calls
              and w.items()[got[0]["approval_id"]].get("execution") is None, r["reasons"])
    finally:
        w.close()

    w = World()
    try:
        w.replay()
        camp = dict(CAMP_ENV, envelope_su=50.0)

        class Crash(BaseException):
            pass

        def crash(script, timeout=60):
            w.fake.calls.append(script)
            raise Crash()

        w.ctx.slurm_sh = crash
        try:
            sub(w, l1(exp="cr_1", est=40.0), campaign=camp)
            check("a lab crash during the ssh call", False, "no crash")
        except Crash:
            pass
        st = X.budget_now(camp, w.ctx)
        check("  leaves the build charged (it may have run)",
              st["committed_su"] == 40.0 and st["remaining_su"] == 10.0, st)
        w.ctx.slurm_sh = w.fake
        r = sub(w, l1(exp="cr_2", est=40.0), campaign=camp)
        check("  so a second 40 SU build does not fit the 50 SU envelope",
              r["status"] == "filed" and any("budget" in x for x in r["reasons"]), r["reasons"])
    finally:
        w.close()

    w = World()
    try:
        w.replay()
        camp = dict(CAMP_ENV, envelope_su=50.0)
        log = w.ctx.exec_log

        def lock_log(script, timeout=60):
            out = Fake.__call__(w.fake, script, timeout)
            os.chmod(str(log), 0o444)                 # the outcome line will not be written
            return out

        w.ctx.slurm_sh = lock_log
        r = sub(w, l1(exp="lk_1", est=40.0), campaign=camp)
        os.chmod(str(log), 0o644)
        st = X.budget_now(camp, w.ctx)
        check("an outcome line that cannot be written is reported, and the run stays charged",
              r["status"] == "executed" and r.get("log_error")
              and any("could not be written" in x for x in r.get("warnings") or [])
              and st["committed_su"] == 40.0, (r.get("warnings"), st["committed_su"]))
    finally:
        w.close()


def section_direct_once():
    print("a direct proposal runs once")
    w = World()
    try:
        p = l3()
        a, b = sub(w, p, campaign=CAMP_OFF), sub(w, p, campaign=CAMP_OFF)
        st = X.budget_now(CAMP_OFF, w.ctx)
        check("the same R2 proposal submitted twice runs once and is charged once",
              a["status"] == "executed" and b["status"] == "refused"
              and "already ran" in b["reasons"][0] and len(w.fake.calls) == 1
              and st["committed_su"] == 1.0, (b["reasons"], len(w.fake.calls), st["committed_su"]))
        p2 = l3()
        out = X.submit_many([p2, p2], campaign=CAMP_OFF, ctx=w.ctx)
        check("  also within one batch", [r["status"] for r in out] == ["executed", "refused"])
        w.fake.mode = "refuse"
        p3 = l3()
        c = sub(w, p3, campaign=CAMP_OFF)
        w.fake.mode = "ok"
        d = sub(w, p3, campaign=CAMP_OFF)
        check("a proposal whose run certainly did not happen (builder refusal) may be retried",
              c["status"] == "failed" and not c["charged"] and d["status"] == "executed",
              (c["status"], d["status"], d["reasons"]))
        raw = X.executions(w.ctx, raw=True)
        check("each run is a started line and an outcome line with one run_id",
              all(r.get("run_id") for r in raw if r.get("status") in ("started", "executed"))
              and len({r["run_id"] for r in raw if r.get("status") == "started"})
              == len([r for r in raw if r.get("status") == "started"])
              and len(X.executions(w.ctx)) < len(raw))
    finally:
        w.close()


def section_cancel_sync():
    print("cancel and sync-outer through the approval path")
    w = World()
    try:
        w.replay()
        c = sub(w, plain("inc_cancel_exp", exp="pilot_v2"), campaign=CAMP_ENV)
        s = sub(w, plain("inc_sync_outer"), campaign=CAMP_ENV)
        check("the autopilot files a cancel and a sync; the envelope does not grant them",
              c["status"] == "filed" and s["status"] == "filed" and not w.fake.calls
              and any("not an envelope build" in x for x in c["reasons"]), c["reasons"])
        for r in (c, s):
            AP.decide("weed", r["approval_id"], "approve", HUMAN, "go", T0 + 70, root=w.dir)
        out = X.run_approved(CAMP_ENV, w.ctx)
        check("  approved by a person, both run as that person in their own verb",
              [x["status"] for x in out] == ["executed", "executed"]
              and all(x["authorized_as"] == HUMAN for x in out)
              and w.fake.verb_lines() == [["cancel", "--exp", "pilot_v2"], ["sync-outer"]],
              w.fake.verb_lines())
        check("render gives remote.py's cancel and sync-outer lines",
              X.render("inc_cancel_exp", {"exp": "x"})["remote"] == ["cancel", "--exp", "x"]
              and X.render("inc_sync_outer", {})["remote"] == ["sync-outer"])
    finally:
        w.close()


def section_unblock_cause():
    print("an automatic unblock needs a transient cause")
    w = World()
    try:
        r = sub(w, l7(unit="base", cause="failed_run"))
        check("a cause off the allow-list is refused, naming the list",
              r["status"] == "refused" and "transient allow-list" in r["reasons"][0]
              and not w.fake.calls, r["reasons"])
        w.fired = [d for d in FIRED if d["id"] != "D5"]
        r = sub(w, l7(unit="chain:full"))
        check("a unit no fired D5 lists as transient is refused",
              r["status"] == "refused" and "no fired D5" in r["reasons"][0], r["reasons"])
        w.fired = list(FIRED)
        r = sub(w, l7(exp="pilot_v9", unit="chain:full"))
        check("  (another experiment's D5 does not count)",
              r["status"] == "refused" and "no fired D5 of pilot_v9" in r["reasons"][0],
              r["reasons"])
        r = sub(w, plain("inc_unblock_transient", exp="pilot_v2", unit="base", cause="failed_run"),
                actor=HUMAN)
        check("a person may unblock for any cause", r["status"] == "executed", r["reasons"])
    finally:
        w.close()
    w = World(diagnoses=None)
    try:
        r = sub(w, l7(unit="chain:full"))
        check("with no diagnosis record the allow-list alone decides", r["status"] == "executed",
              r["reasons"])
    finally:
        w.close()


def section_resources_and_caps():
    print("unknown resources and the domain caps")
    w = World(resources=None)
    try:
        r = sub(w, l3(), campaign=CAMP_OFF)
        check("no resources hook: a GPU action is refused (unknown Mongo is not healthy)",
              r["status"] == "refused" and "Mongo's health" in r["reasons"][0]
              and not w.fake.calls, r["reasons"])
        r = sub(w, plain("inc_snapshot", exp="pilot_v1"), campaign=CAMP_OFF)
        check("  a free read still runs", r["status"] == "executed")
        r = sub(w, l1(), campaign=CAMP_OFF)
        check("  and an R3 build is still filed (filing spends nothing)", r["status"] == "filed")
        w.replay()
        r = sub(w, l1(exp="pilot_m1", est=5.0), campaign=CAMP_ENV)
        check("  but not granted", r["status"] == "filed"
              and any("Mongo's health" in x for x in r["reasons"]), r["reasons"])
        st = X.budget_now(dict(CAMP_OFF, daily_cap_su=1000.0), w.ctx)
        check("a campaign daily cap is capped at the domain daily_cap",
              st["daily_cap_su"] == 120.0, st["daily_cap_su"])
        env = B.envelope(dict(CAMP_OFF, daily_cap_su=-5.0))
        check("  a negative one is 0", env["daily_cap_su"] == 0.0, env)
    finally:
        w.close()


def section_retries_and_locks():
    print("retries are quiet; decide takes the log lock")
    w = World()
    try:
        w.spend("old", 295.0)
        r = sub(w, l1(est=40.0), campaign=CAMP_OFF)
        AP.decide("weed", r["approval_id"], "approve", HUMAN, "go", T0 + 5, root=w.dir)
        for _ in range(5):
            X.run_approved(CAMP_OFF, w.ctx)
        n = sum(1 for x in X.executions(w.ctx) if x.get("approval_id") == r["approval_id"]
                and x["status"] == "refused")
        check("an approved item that cannot run yet logs its refusal once, not every tick",
              n == 1, n)
        rec = [{"campaign": "c", "action": "inc_build_pilot", "params": {"exp": "new_x"},
                "child_exp": "settled_y", "charged": True, "est_su": 10.0, "epoch": T0}]
        check("the budget releases a build only by its own --exp, whatever child_exp says",
              B.committed(rec, "c", ["settled_y"])["su"] == 10.0
              and B.committed(rec, "c", ["new_x"])["su"] == 0.0)
    finally:
        w.close()
    w = World()
    try:
        item = AP.propose("weed", "inc_build_pilot", {"exp": "p"}, "R3", AUTO, "why", T0,
                          root=w.dir)["item"]
        out = []
        lock = AP._LogLock(AP.path("weed", root=w.dir))
        lock.__enter__()
        t = threading.Thread(target=lambda: out.append(
            AP.decide("weed", item["id"], "deny", HUMAN, "no", T0 + 1, root=w.dir)))
        t.start()
        time.sleep(0.3)
        waited = not out
        lock.__exit__(None, None, None)
        t.join(5)
        check("decide waits for the approval log's lock", waited and out and out[0]["ok"], out)
        check("decide on a domain with no log creates nothing",
              not AP.decide("nodomain", "x", "approve", HUMAN, "r", T0, root=w.dir)["ok"]
              and not os.path.exists(os.path.dirname(AP.path("nodomain", root=w.dir))))
    finally:
        w.close()


def section_replay_gate():
    print("replay not passed blocks autonomy")
    for label, setup, needle in (
            ("no replay result", lambda w: None, "no replay result"),
            ("a failed replay", lambda w: w.replay("fail"), "not pass"),
            ("a pass on other code", lambda w: w.replay("pass", code="0" * 64), "other autopilot"),
            ("an unreadable result", lambda w: (w.ctx.replay_result.parent.mkdir(parents=True),
                                                w.ctx.replay_result.write_text("{torn")),
             "unreadable")):
        w = World()
        try:
            setup(w)
            r = sub(w, l1(), campaign=CAMP_ENV)
            check("%s: the build is filed, not run" % label,
                  r["status"] == "filed" and not w.fake.calls
                  and any(needle in x for x in r["reasons"]), r["reasons"])
        finally:
            w.close()
    w = World()
    try:
        w.replay()
        st = X.replay_status(w.ctx)
        check("a pass recorded on the current code counts", st["passed"], st["reason"])
        r = sub(w, l1(), campaign=CAMP_OFF)
        item_id = r["approval_id"]
        AP.approve_within_envelope("weed", item_id, AUTO,
                                   {"lever": "L1", "trigger": ["D1"], "cites": CITE,
                                    "authority": OWNER,
                                    "envelope": {"remaining_su": 300.0, "est_su": 40.0}},
                                   "granted", T0 + 50, root=w.dir)
        w.replay("fail")
        r2 = X.execute_approved(item_id, CAMP_ENV, w.ctx)
        check("the envelope rule is re-checked at execution: a replay gone stale refuses",
              r2["status"] == "refused" and not w.fake.calls
              and w.items()[item_id].get("execution") is None, r2["reasons"])
        w.replay("pass")
        r3 = X.execute_approved(item_id, CAMP_ENV, w.ctx)
        check("  and with the pass back it runs once", r3["status"] == "executed", r3["reasons"])
    finally:
        w.close()
    tmp = tempfile.mkdtemp(prefix="inc_ap_hash_")
    try:
        (pathlib.Path(tmp) / "a.py").write_text("x = 1\n")
        (pathlib.Path(tmp) / "t.json").write_text("{}")
        h1 = X.code_hash(tmp)
        (pathlib.Path(tmp) / "t.json").write_text('{"threshold": 2}')
        check("the code hash changes when a threshold file changes", X.code_hash(tmp) != h1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    check("the hash covers the policy table and gate, the approval queue, the SU ledger and "
          "both test scripts",
          {"weed_optimizer_framework/tools/brain/policy_actions.json",
           "weed_optimizer_framework/tools/brain/policy.py",
           "weed_optimizer_framework/tools/brain/approvals.py",
           "weed_optimizer_framework/tools/brain/su_ledger.py", "tests/test_inc_ap_replay.py",
           "tests/test_inc_ap_governance.py"} <= set(X.GOVERNANCE_FILES))
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="inc_ap_gov_root_"))
    old_root = X.CODE_ROOT
    try:
        for rel in X.GOVERNANCE_FILES:
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp / rel).write_text("v1\n")
        X.CODE_ROOT = tmp
        h1 = X.code_hash()
        for rel in X.GOVERNANCE_FILES:
            (tmp / rel).write_text("v2\n")
            h2 = X.code_hash()
            check("  changing %s cancels a recorded pass" % rel.rsplit("/", 1)[-1], h2 != h1)
            h1 = h2
        (tmp / X.GOVERNANCE_FILES[0]).unlink()
        check("  a missing governance file changes the hash too", X.code_hash() != h1)
    finally:
        X.CODE_ROOT = old_root
        shutil.rmtree(str(tmp), ignore_errors=True)

    w = World()
    try:
        for cases, needle in (({}, "R1"), (dict(REPLAY_PASS, R3="skip"), "R3"),
                              (dict(REPLAY_PASS, test_blindness="skip"), "test_blindness"),
                              ({k: v for k, v in REPLAY_PASS.items() if k != "R2"}, "R2"),
                              (dict(REPLAY_PASS, extra="fail"), "extra")):
            try:
                X.record_replay_result("pass", cases, ctx=w.ctx)
                check("a pass missing %s is not recorded" % needle, False)
            except ValueError as e:
                check("a pass missing %s is not recorded" % needle, needle in str(e), str(e))
        w.ctx.replay_result.parent.mkdir(parents=True, exist_ok=True)
        w.ctx.replay_result.write_text(json.dumps({"status": "pass", "code_hash": X.code_hash(),
                                                   "cases": {}, "recorded_utc": "x"}))
        st = X.replay_status(w.ctx)
        check("a pass written with no cases does not count",
              not st["passed"] and "incomplete" in st["reason"], st["reason"])
        r = sub(w, l1(exp="pilot_e1", est=5.0), campaign=CAMP_ENV)
        check("  and an envelope build waits for a person",
              r["status"] == "filed" and not w.fake.calls, r["reasons"])
        w.replay()
        check("a pass with every required case and R2/R4b skipped counts",
              X.replay_status(w.ctx)["passed"])
    finally:
        w.close()

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="inc_ap_gov_run_"))
    w = World()
    try:
        def script(name, out, rc):
            (tmp / name).write_text("import sys\nprint(%r)\nsys.exit(%d)\n" % (out, rc))

        runs = (("clean", "0 failure(s), 2 skipped: R2, R4b on pilot_v2", 0, "ALL PASS", 0, "pass",
                 {"R2": "skip", "R4b": "skip", "R1": "pass", "governance": "pass"}),
                ("replay fails", "3 failure(s), 0 skipped: none", 1, "ALL PASS", 0, "fail",
                 {"R1": "fail", "governance": "pass"}),
                ("an unknown skip", "0 failure(s), 1 skipped: R3 partial", 0, "ALL PASS", 0, "fail",
                 {"unrecognised_skip": "fail"}),
                ("governance fails", "0 failure(s), 0 skipped: none", 0, "1 failure(s)", 1, "fail",
                 {"R2": "pass", "governance": "fail"}),
                ("no summary line", "done", 0, "ALL PASS", 0, "fail", {"R1": "fail"}))
        # The funnel audit's replay, mutation and domain-free scripts pass here
        # (their own cases are pinned in tests/test_funnel_ap_units.py).
        for name in ("f.py", "m.py", "d.py"):
            script(name, "0 failure(s), 0 skipped: none", 0)
        for label, rout, rrc, gout, grc, want, sub_cases in runs:
            script("r.py", rout, rrc)
            script("g.py", gout, grc)
            rec = X.run_replay_tests(ctx=w.ctx, scripts={"replay": "r.py", "governance": "g.py", "funnel": "f.py",
                                                         "funnel_mutations": "m.py", "domain_free": "d.py"},
                                     code_root=tmp)
            got = {k: rec["cases"].get(k) for k in sub_cases}
            check("replay runner, %s: records %s with %s" % (label, want, sub_cases),
                  rec["status"] == want and got == sub_cases
                  and X.replay_status(w.ctx)["passed"] is (want == "pass"), (rec["status"], got))
    finally:
        w.close()
        shutil.rmtree(str(tmp), ignore_errors=True)


def section_budget():
    print("budget exhaustion blocks")
    w = World()
    try:
        w.replay()
        w.spend("old_exp", 280.0)
        st = X.budget_now(CAMP_ENV, w.ctx)
        check("the envelope is the 300 SU default less the ledger's spend",
              st["envelope_su"] == 300.0 and st["spent_su"] == 280.0 and st["remaining_su"] == 20.0,
              st)
        r = sub(w, l1(est=40.0), campaign=CAMP_ENV)
        check("an L1 build over the balance is not self-approved",
              r["status"] == "filed" and any("budget" in x for x in r["reasons"]), r["reasons"])
        AP.decide("weed", r["approval_id"], "approve", HUMAN, "go", T0 + 100, root=w.dir)
        r2 = X.execute_approved(r["approval_id"], CAMP_ENV, w.ctx)
        check("a person's approval does not run it past the envelope either",
              r2["status"] == "refused" and "budget" in " ".join(r2["reasons"])
              and not w.fake.calls, r2["reasons"])
        check("  and the item stays approved and unexecuted",
              w.items()[r["approval_id"]].get("execution") is None)
        r3 = sub(w, plain("inc_relevance_build"), campaign=CAMP_ENV)
        check("a 1 SU relevance build still fits", r3["status"] == "executed", r3["reasons"])
        w.spend("old_exp2", 19.5)
        r4 = sub(w, plain("inc_relevance_build", sample=500), campaign=CAMP_ENV)
        check("with 0.5 SU left the next one is refused, not filed",
              r4["status"] == "refused" and "budget" in r4["reasons"][0], r4["reasons"])
        r5 = sub(w, plain("inc_snapshot", exp="pilot_v1"), campaign=CAMP_ENV)
        check("free reads still run on an exhausted envelope", r5["status"] == "executed")
    finally:
        w.close()

    w = World()
    try:
        w.replay()
        camp = dict(CAMP_ENV, daily_cap_su=50.0)
        a = sub(w, l1(exp="pilot_v2", est=40.0), campaign=camp)
        b = sub(w, l1(exp="pilot_v3", est=40.0), campaign=camp)
        check("the daily cap stops a second 40 SU build the same day",
              a["status"] == "executed" and b["status"] == "filed"
              and any("today" in x for x in b["reasons"]), (a["reasons"], b["reasons"]))
        w.tick(86400)
        again = X.submit(l1(exp="pilot_v3", est=40.0, id=b["proposal_id"]), campaign=camp,
                         ctx=w.ctx)
        check("  the next day the same proposal runs from its filed item",
              again["status"] == "executed" and again["approval_id"] == b["approval_id"],
              again["reasons"])
        st = X.budget_now(camp, w.ctx)
        check("both estimates are committed while no report has landed",
              st["committed_su"] == 80.0 and st["remaining_su"] == 220.0, st)
        B.record_report_spend({"exp": "pilot_v2", "done": True, "done_utc": "2026-09-22T00:00:00Z",
                               "gpu_hours": {"truth": {"hours": 5.0, "runs": 21},
                                             "chain:full": {"hours": 2.0, "runs": 42}}},
                              "weedinc1", base_dir=w.ctx.su_base_dir)
        st = X.budget_now(camp, w.ctx)
        check("a finished report replaces its estimate with the measured spend",
              st["spent_su"] == 7.0 and st["committed_su"] == 40.0 and st["remaining_su"] == 253.0,
              st)
    finally:
        w.close()

    w = World()
    try:
        r = sub(w, plain("inc_relevance_build"), campaign=None)
        check("a GPU action with no campaign envelope is refused",
              r["status"] == "refused" and "envelope" in r["reasons"][0], r["reasons"])
        r = sub(w, plain("inc_snapshot", exp="pilot_v1"), campaign=None)
        check("a free read needs no campaign", r["status"] == "executed")
        r = sub(w, plain("inc_build_pilot", exp="p", replay_mode="full"), actor=HUMAN)
        check("a build with no estimate is refused, even for a person",
              r["status"] == "refused" and "unknown" in r["reasons"][0], r["reasons"])
        big = dict(CAMP_OFF, envelope_su=5000.0)
        st = X.budget_now(big, w.ctx)
        check("a campaign envelope is capped at the domain su_envelope",
              st["envelope_su"] == 1500.0, st)
    finally:
        w.close()


def section_idempotency():
    print("one approval id runs once")
    w = World()
    try:
        r = sub(w, l1(), campaign=CAMP_OFF)
        again = sub(w, l1(id=r["proposal_id"]), campaign=CAMP_OFF)
        reqs = [x for x in AP.read("weed", root=w.dir) if x.get("kind") == "request"]
        check("resubmitting the same proposal files nothing new",
              again["approval_id"] == r["approval_id"] and len(reqs) == 1, again["reasons"])
        AP.decide("weed", r["approval_id"], "approve", HUMAN, "go ahead", T0 + 10, root=w.dir)
        ran = X.run_approved(CAMP_OFF, w.ctx)
        check("run_approved runs the approved item", [x["status"] for x in ran] == ["executed"])
        check("  authorised as the person who approved it", ran[0]["authorized_as"] == HUMAN)
        check("run_approved has nothing left to run", X.run_approved(CAMP_OFF, w.ctx) == [])
        r2 = X.execute_approved(r["approval_id"], CAMP_OFF, w.ctx)
        check("a second execution of the same approval id is refused",
              r2["status"] == "refused" and "already executed" in r2["reasons"][0], r2["reasons"])
        check("  and the cluster was called once", len(w.fake.calls) == 1)
        ex = w.items()[r["approval_id"]]["execution"]
        check("the item records its execution", ex["phase"] == "done"
              and ex["executed_by"] == AUTO and ex["outcome"]["authorized_as"] == HUMAN, ex)
        check("record_executed refuses a second start",
              not AP.record_executed("weed", r["approval_id"], "started", AUTO, T0 + 20,
                                     root=w.dir)["ok"])
        with open(AP.path("weed", root=w.dir), "a") as fh:
            fh.write(json.dumps({"kind": "executed", "id": r["approval_id"], "phase": "started",
                                 "executed_by": "tier1:x", "ts": T0 + 21}) + "\n")
        item = w.items()[r["approval_id"]]
        check("a forged second start in the log is kept as an attempt, not applied",
              item["execution"]["phase"] == "done" and len(item["execution_attempts"]) == 1)
    finally:
        w.close()

    w = World()
    try:
        r = sub(w, l1(), campaign=CAMP_OFF)
        AP.decide("weed", r["approval_id"], "approve", HUMAN, "go", T0 + 10, root=w.dir)
        w.fake.delay = 0.3
        out = []
        ths = [threading.Thread(target=lambda: out.append(
            X.execute_approved(r["approval_id"], CAMP_OFF, w.ctx)["status"])) for _ in range(2)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        check("two racing executions of one approval: one runs, one is refused",
              sorted(out) == ["executed", "refused"] and len(w.fake.calls) == 1, out)
    finally:
        w.close()

    w = World()
    try:
        r = sub(w, l1(), campaign=CAMP_OFF)
        AP.decide("weed", r["approval_id"], "approve", HUMAN, "go", T0 + 10, root=w.dir)
        w.fake.mode = "refuse"
        r2 = X.execute_approved(r["approval_id"], CAMP_OFF, w.ctx)
        check("a builder refusal marks the execution failed and charges nothing",
              r2["status"] == "failed" and not r2["charged"]
              and w.items()[r["approval_id"]]["execution"]["phase"] == "failed", r2["reasons"])
        w.fake.mode = "ok"
        r3 = X.execute_approved(r["approval_id"], CAMP_OFF, w.ctx)
        check("  and a failed approval is not retried", r3["status"] == "refused")
        d = sub(w, l1(exp="pilot_v9"), campaign=CAMP_OFF)
        AP.decide("weed", d["approval_id"], "deny", HUMAN, "not now", T0 + 30, root=w.dir)
        r4 = X.execute_approved(d["approval_id"], CAMP_OFF, w.ctx)
        check("a denied item never runs", r4["status"] == "refused" and "denied" in r4["reasons"][0])
        r5 = X.execute_approved(r["approval_id"], dict(CAMP_OFF, name="other"), w.ctx)
        check("an item filed for one campaign does not run under another",
              r5["status"] == "refused")
    finally:
        w.close()


def section_r4():
    print("R4 is never executed")
    w = World()
    try:
        card = {"lever": "X1", "risk": "R4", "policy_action": None,
                "summary": "LR re-warm needs a pinned-module change"}
        for actor in (AUTO, HUMAN):
            r = sub(w, card, actor=actor)
            check("an off-menu R4 card is refused for %s" % actor,
                  r["status"] == "refused" and "human card" in r["reasons"][0], r["reasons"])
        r = sub(w, l1(risk="R2"))
        check("a request whose risk disagrees with the table is refused",
              r["status"] == "refused" and "policy table says R3" in r["reasons"][0])
        table = json.loads((REPO / "weed_optimizer_framework/tools/brain/policy_actions.json")
                           .read_text())
        table["actions"]["inc_repin"] = {
            "template": "python -m weed_optimizer_framework.tools.inc.driver repin --exp {exp}",
            "param_bounds": {"exp": {"type": "str", "pattern": "^[A-Za-z0-9_-]{1,64}$"}},
            "risk": "R4", "reversible": False, "est_su": {"fixed_su": 0.0, "why": "test row"},
            "dry_run_variant": None, "allowed_tiers": ["round-scheduler", "human"],
            "description": "test-only R4 row"}
        tpath = pathlib.Path(w.dir) / "policy_actions.json"
        tpath.write_text(json.dumps(table))
        old = os.environ.get("BRAIN_POLICY_ACTIONS")
        os.environ["BRAIN_POLICY_ACTIONS"] = str(tpath)
        try:
            check("the test table has an R4 row", POL.risk_of("inc_repin") == "R4")
            check("policy itself lets a person take R4 directly",
                  POL.authorize(HUMAN, "inc_repin", {"exp": "pilot_v2"})["allowed"])
            for actor in (AUTO, HUMAN):
                r = sub(w, plain("inc_repin", exp="pilot_v2"), actor=actor)
                check("the executor refuses an R4 row for %s" % actor,
                      r["status"] == "refused" and "R4" in r["reasons"][0] and not w.fake.calls,
                      r["reasons"])
            check("the approval queue refuses to hold R4",
                  not AP.propose("weed", "inc_repin", {"exp": "p"}, "R4", HUMAN, "why", T0,
                                 root=w.dir)["ok"])
        finally:
            if old is None:
                os.environ.pop("BRAIN_POLICY_ACTIONS", None)
            else:
                os.environ["BRAIN_POLICY_ACTIONS"] = old
        check("the real table is back", POL.describe("inc_repin")["known"] is False)
    finally:
        w.close()


def section_rendering():
    print("what runs is what the policy checked")
    p1 = l1()
    check("L1 renders to the manual command",
          X.render("inc_build_pilot", {"exp": "pilot_v2", "replay_mode": "full"})["builder"]
          == ["inc.pilot", "build", "--exp", "pilot_v2", "--replay-mode", "full"])
    check("L1's parameters come out of its argv",
          X.params_from_argv("inc_build_pilot", p1["argv"]) == {"exp": "pilot_v2",
                                                                "replay_mode": "full"})
    toks = audit_command()
    exp = json.loads((PILOT / "exp.json").read_text())
    by_name = {s["name"]: s["manifest"] for s in exp["steps"]}
    names = toks[toks.index("--audit") + 1:toks.index("--out")]
    params = {"trusted": exp["base"]["manifest"],
              "audit": ",".join("%s=%s" % (n.split("=")[0], by_name[n.split("=")[0]])
                                for n in names),
              "out": toks[toks.index("--out") + 1]}
    check("L4 from pilot_v1's exp.json renders run_inc_audit.sh's documented command",
          X.render("inc_label_audit", params)["builder"] == toks,
          (X.render("inc_label_audit", params)["builder"], toks))
    check("  and that command parses back to the same parameters",
          X.params_from_argv("inc_label_audit", ["sbatch"] + toks) == params)
    check("  and the policy allows it", POL.authorize(AUTO, "inc_label_audit", params)["allowed"])
    full = {"exp": "real_v1", "base": INC + "step1/base_B.jsonl", "replay_mode": "sample",
            "recipes": "full,freeze", "relevance": INC + "step1/relevance.json", "size": 300,
            "n_verified": 6, "no_truth": 1}
    b = X.render("inc_build_realloop", full)["builder"]
    check("a realloop build round-trips through its argv",
          X.params_from_argv("inc_build_realloop", b) == full and b[-1] == "--no-truth", b)
    check("the unblock reason is 'auto: <cause>'",
          X.render("inc_unblock_transient", {"exp": "p", "unit": "truth", "cause": "transient"})
          ["remote"] == ["unblock", "--exp", "p", "--unit", "truth", "--reason", "auto: transient"])
    check("recipes are put in the table's order", X.canonical_recipes(["lora", "full"]) == "full,lora")
    l9 = {"exp": "pilot_v3", "replay_mode": "full", "gate_flips_mode": "net"}
    b = X.render("inc_build_pilot", l9)["builder"]
    check("L9 renders --gate-flips-mode last and round-trips through its argv",
          b == ["inc.pilot", "build", "--exp", "pilot_v3", "--replay-mode", "full", "--gate-flips-mode", "net"]
          and X.params_from_argv("inc_build_pilot", b) == l9, b)
    b = X.render("inc_build_realloop", dict(full, gate_flips_mode="net"))["builder"]
    check("a realloop build with the v2 gate round-trips, the flag after --no-truth",
          b[-3:] == ["--no-truth", "--gate-flips-mode", "net"]
          and X.params_from_argv("inc_build_realloop", b) == dict(full, gate_flips_mode="net"), b)
    check("the envelope covers L9 (inc_build_pilot) and checks it against L9's own bounds",
          X.ENVELOPE_LEVERS.get("L9") == ("inc_build_pilot",)
          and not X._lever_params_check("L9", dict(l9, est_gpu_hours=31.2))
          and X._lever_params_check("L9", dict(l9, gate_flips_mode="negative", est_gpu_hours=31.2)), l9)
    check("the policy allows an L9 build for a person and files it for the autopilot",
          POL.authorize("human:owner@example.org", "inc_build_pilot", dict(l9, est_gpu_hours=31.2))["allowed"]
          and not POL.authorize("human:owner@example.org", "inc_build_pilot",
                                dict(l9, gate_flips_mode="both", est_gpu_hours=31.2))["allowed"])
    check("a replay pass needs R5 (pilot_v2's case, L9's replay)", "R5" in X.REPLAY_REQUIRED)
    check("a replay pass needs R6 (pilot_v3's case: D1 does not block D4, L2 on gate net)",
          "R6" in X.REPLAY_REQUIRED
          and X.check_replay_cases(dict({c: "pass" for c in X.REPLAY_REQUIRED if c != "R6"}, R2="skip", R4b="skip"))
          == ["case R6 is None, not pass"])
    check("a replay pass needs R7 (a calibration-failed relevance.json: L2 on --increment-sources evidence)",
          "R7" in X.REPLAY_REQUIRED
          and X.check_replay_cases(dict({c: "pass" for c in X.REPLAY_REQUIRED if c != "R7"}, R2="skip", R4b="skip"))
          == ["case R7 is None, not pass"])
    inc = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc"
    l2e = {"exp": "realloop_v1", "base": inc + "/step1/base_B.jsonl", "replay_mode": "full", "recipes": "full",
           "increment_sources": "evidence", "gate_flips_mode": "net", "est_gpu_hours": 120.0}
    check("an evidence L2 (R7's) is inside L2's own bounds (the envelope's check) and the policy row's; a person "
          "may build it, and an increment_sources outside the enum is refused",
          not X._lever_params_check("L2", l2e)
          and POL.authorize("human:owner@example.org", "inc_build_realloop", l2e)["allowed"]
          and not POL.authorize("human:owner@example.org", "inc_build_realloop",
                                dict(l2e, increment_sources="hand_list"))["allowed"]
          and X._lever_params_check("L2", dict(l2e, increment_sources="hand_list")), l2e)
    r = X.render("inc_build_realloop", {k: v for k, v in l2e.items() if k != "est_gpu_hours"})
    check("the executor renders it in levers.py's order (--increment-sources before --gate-flips-mode) and reads "
          "the argv back to the same params",
          r["builder"][-4:] == ["--increment-sources", "evidence", "--gate-flips-mode", "net"]
          and X.params_from_argv("inc_build_realloop", r["builder"]) == {k: v for k, v in l2e.items()
                                                                        if k != "est_gpu_hours"}, r["builder"])
    check("a replay pass needs R8 (the real Step 1: the R4 sizing rule, L2 at N 4 x M 287)",
          "R8" in X.REPLAY_REQUIRED
          and X.check_replay_cases(dict({c: "pass" for c in X.REPLAY_REQUIRED if c != "R8"}, R2="skip", R4b="skip"))
          == ["case R8 is None, not pass"])
    l2s = dict(l2e, size=287, n_verified=4)
    check("a sized evidence L2 (R8's: --size 287 --n-verified 4) is inside L2's own bounds and the policy row's; a "
          "person may build it; an n_verified or size outside L2's bounds is refused by the envelope's check",
          not X._lever_params_check("L2", l2s)
          and POL.authorize("human:owner@example.org", "inc_build_realloop", l2s)["allowed"]
          and X._lever_params_check("L2", dict(l2s, n_verified=13))
          and X._lever_params_check("L2", dict(l2s, size=0)), l2s)
    r = X.render("inc_build_realloop", {k: v for k, v in l2s.items() if k != "est_gpu_hours"})
    check("the executor renders it in levers.py's order (--increment-sources evidence --size 287 --n-verified 4 "
          "--gate-flips-mode net) and reads the argv back to the same params",
          r["builder"][-8:] == ["--increment-sources", "evidence", "--size", "287", "--n-verified", "4",
                                "--gate-flips-mode", "net"]
          and X.params_from_argv("inc_build_realloop", r["builder"]) == {k: v for k, v in l2s.items()
                                                                        if k != "est_gpu_hours"}, r["builder"])
    pair = dict({k: v for k, v in l2e.items() if k != "est_gpu_hours"}, relevance=inc + "/step1/relevance.json")
    pair_argv = (["python", "-m", "weed_optimizer_framework.tools.inc.realloop", "build", "--exp", "realloop_v1",
                  "--base", inc + "/step1/base_B.jsonl", "--replay-mode", "full", "--recipes", "full",
                  "--increment-sources", "evidence", "--relevance", inc + "/step1/relevance.json",
                  "--gate-flips-mode", "net"])
    row = POL.describe("inc_build_realloop")

    def exec_error(fn):
        try:
            fn()
        except X.ExecError as e:
            return str(e)
        return ""
    check("--increment-sources evidence with --relevance is refused on the lab side with remote.py's message: "
          "executor.render, resolve_params (params or argv), the envelope's lever check (levers.check_params)",
          exec_error(lambda: X.render("inc_build_realloop", pair)) == M.EVIDENCE_WITH_RELEVANCE
          and exec_error(lambda: X.resolve_params("inc_build_realloop", row, pair, None, 120.0))
          == M.EVIDENCE_WITH_RELEVANCE
          and exec_error(lambda: X.resolve_params("inc_build_realloop", row, {}, pair_argv, 120.0))
          == M.EVIDENCE_WITH_RELEVANCE
          and M.EVIDENCE_WITH_RELEVANCE in "; ".join(X._lever_params_check("L2", dict(pair, est_gpu_hours=120.0)))
          and "does not go with it" in M.EVIDENCE_WITH_RELEVANCE,
          (exec_error(lambda: X.render("inc_build_realloop", pair)),
           X._lever_params_check("L2", dict(pair, est_gpu_hours=120.0))))
    check("... --increment-sources relevance with --relevance (the builders' default pair) still renders",
          X.render("inc_build_realloop", dict(pair, increment_sources="relevance"))["builder"][-6:-2]
          == ["--increment-sources", "relevance", "--relevance", inc + "/step1/relevance.json"])
    w = World()
    try:
        bad = l1()
        bad["params"] = {"exp": "pilot_v2", "replay_mode": "sample"}
        r = sub(w, bad)
        check("params that contradict the argv are refused",
              r["status"] == "refused" and "argv says" in r["reasons"][0], r["reasons"])
        r = sub(w, l1(argv=l1()["argv"] + ["--force"]))
        check("a flag the grammar does not know is refused",
              r["status"] == "refused" and "--force" in r["reasons"][0], r["reasons"])
        swapped = ["python", "-m", "weed_optimizer_framework.tools.inc.pilot", "build",
                   "--replay-mode", "full", "--exp", "pilot_v2"]
        r = sub(w, l1(argv=swapped))
        check("an argv the executor would not render byte for byte is refused",
              r["status"] == "refused" and "not the command" in r["reasons"][0], r["reasons"])
        a = M.proposal("L4", ["sbatch", "run_inc_audit.sh", "--trusted", "$M/P0.jsonl", "--audit",
                              "I1=$M/I1.jsonl", "--out", "$INC/pilot_v1/audit/a.json"],
                       {"exp": "pilot_v1"}, "inc_label_audit", "R2", ["D3"], CITE)
        r = sub(w, a, campaign=CAMP_OFF)
        check("an argv with unexpanded shell variables is refused",
              r["status"] == "refused" and "pattern" in " ".join(r["reasons"]), r["reasons"])
        r = sub(w, dict(l1(), params={"exp": "pilot_v2", "est_gpu_hours": 30.0}))
        check("two different prices are refused", r["status"] == "refused"
              and "two prices" in r["reasons"][0], r["reasons"])
        r = sub(w, l1(proposed_by=BRAIN), actor=AUTO)
        check("a brain proposal cannot borrow the autopilot's authority",
              r["status"] == "refused" and "proposed by" in r["reasons"][0], r["reasons"])
        r = sub(w, l3())
        check("L3 runs with its price kept as metadata (the row prices by walltime)",
              r["status"] == "executed" and r["params"] == {}
              and r["meta_params"] == {"est_gpu_hours": 1.0} and r["est_su"] == 1.0, r)
        r = sub(w, plain("inc_snapshot", exp="pilot_v1", force=1))
        check("without an argv an undeclared parameter is refused by the policy",
              r["status"] == "refused" and "no declared bound" in r["reasons"][0], r["reasons"])
    finally:
        w.close()
    line = X.render("inc_build_pilot", {"exp": "pilot_v2", "replay_mode": "full"},
                    {"parent_exp": "pilot_v1", "trigger": "D1,D4", "approval_id": "ap-1-2",
                     "decided_by": HUMAN})["remote"]
    builder, meta, dry, args = R._split_submit(line[1:])
    req = R.parse_builder_args(builder, args)
    check("remote.py reads the submit line as the same builder, provenance and parameters",
          builder == "build" and not dry and req["action"] == "inc_build_pilot"
          and req["params"] == {"exp": "pilot_v2", "replay_mode": "full"}
          and meta == {"parent_exp": "pilot_v1", "trigger": "D1,D4", "approval_id": "ap-1-2",
                       "decided_by": HUMAN}, (builder, meta, req))
    line = X.render("inc_relevance_build", {"sample": 300, "seed": 0})["remote"]
    builder, meta, dry, args = R._split_submit(line[1:])
    check("  likewise for a relevance build",
          R.parse_builder_args(builder, args)["action"] == "inc_relevance_build")


def section_remote():
    print("remote plumbing")
    w = World()
    try:
        out = X.submit_many([plain("inc_snapshot", exp="pilot_v1"), plain("inc_advance", exp="pilot_v1"),
                             plain("inc_report", exp="pilot_v1")], campaign=CAMP_OFF, ctx=w.ctx)
        check("three direct verbs share one ssh call", len(w.fake.calls) == 1
              and [r["status"] for r in out] == ["executed"] * 3, [r["reasons"] for r in out])
        check("  each gets its own INCAP record, login noise ignored",
              [r["remote"]["payload"]["verb"] for r in out] == ["snapshot", "advance", "report"])
        cs = X.campaign_snapshot(["pilot_v1", "pilot_v2"], advance=True, ledger_from={"pilot_v1": 40},
                                 campaign=CAMP_OFF, ctx=w.ctx)
        line = w.fake.verb_lines()[-1]
        check("campaign-snapshot runs as one verb after authorising each part",
              cs["status"] == "executed" and line == ["campaign-snapshot", "--exp", "pilot_v1",
                                                      "--exp", "pilot_v2", "--advance", "--report",
                                                      "auto", "--ledger-from", "pilot_v1=40"], line)
        cs = X.campaign_snapshot(["pilot_v1"], advance=True, actor=BRAIN, campaign=CAMP_OFF,
                                 ctx=w.ctx)
        check("  a brain may not advance through it", cs["status"] == "refused", cs["reasons"])
        paused = dict(CAMP_OFF, paused_reason="stop-loss")
        check("  a paused campaign reads but does not advance",
              X.campaign_snapshot(["pilot_v1"], advance=True, campaign=paused, ctx=w.ctx)["status"]
              == "refused"
              and X.campaign_snapshot(["pilot_v1"], campaign=paused, ctx=w.ctx)["status"]
              == "executed")
    finally:
        w.close()
    for mode, charged, needle in (("preamble", False, "preamble"), ("raise", False, "never started"),
                                  ("silent", True, "outcome is unknown")):
        w = World()
        try:
            w.fake.mode = mode
            r = sub(w, plain("inc_relevance_build"), campaign=CAMP_OFF)
            check("%s: the action fails%s" % (mode, " and is charged (it may have run)"
                                              if charged else " and is not charged"),
                  r["status"] == "failed" and r["charged"] is charged
                  and needle in " ".join(r["reasons"]), (r["status"], r["charged"], r["reasons"]))
            st = X.budget_now(CAMP_OFF, w.ctx)
            check("  the envelope %s its estimate" % ("keeps" if charged else "does not keep"),
                  st["committed_su"] == (1.0 if charged else 0.0), st["committed_su"])
        finally:
            w.close()
    w = World()
    try:
        ctx = X.Context(slurm_sh=None, lab_repo=w.dir, clock=lambda: T0)
        r = X.submit(plain("inc_snapshot", exp="pilot_v1"), campaign=CAMP_OFF, ctx=ctx)
        check("no slurm_sh hook: refused", r["status"] == "refused" and "slurm_sh" in r["reasons"][0])
        ctx = X.Context(slurm_sh=w.fake, lab_repo=w.dir, clock=lambda: T0,
                        resources={"cluster_reachable": False})
        r = X.submit(plain("inc_snapshot", exp="pilot_v1"), campaign=CAMP_OFF, ctx=ctx)
        check("an unreachable cluster refuses before any call",
              r["status"] == "refused" and not w.fake.calls)
        ctx = X.Context(slurm_sh=w.fake, lab_repo=w.dir, clock=lambda: T0,
                        resources=lambda: {"mongo_ok": False})
        r1 = X.submit(plain("inc_advance", exp="pilot_v1"), campaign=CAMP_OFF, ctx=ctx)
        r0 = X.submit(plain("inc_snapshot", exp="pilot_v1"), campaign=CAMP_OFF, ctx=ctx)
        check("mongo down: R1 is refused by the policy's resource check, R0 still reads",
              r1["status"] == "refused" and "mongo_down" in r1["reasons"][0]
              and r0["status"] == "executed", (r1["reasons"], r0["status"]))
        lit = plain("inc_lit_fetch", arxiv_id="2403.08763", paper_id="ibrahim2024")
        r = X.submit(lit, campaign=CAMP_OFF, ctx=ctx)
        check("a lab action with no hook is refused", r["status"] == "refused")
        seen = []
        ctx = X.Context(slurm_sh=w.fake, lab_repo=w.dir, clock=lambda: T0,
                        local_hooks={"inc_lit_fetch": lambda p: seen.append(p) or {"ok": True}})
        r = X.submit(lit, campaign=CAMP_OFF, ctx=ctx)
        check("a lab action runs its hook with the checked parameters",
              r["status"] == "executed" and seen == [lit["params"]], r["reasons"])
        log = X.executions(w.ctx)
        check("every call left one line in executions.jsonl", len(log) >= 6
              and all(x.get("status") for x in log))
    finally:
        w.close()


def section_unblock_once():
    print("at most one automatic unblock per unit")
    w = World()
    try:
        a = sub(w, l7(unit="chain:full"))
        b = sub(w, l7(unit="chain:full"))
        c = sub(w, l7(unit="truth"))
        d = sub(w, plain("inc_unblock_transient", exp="pilot_v2", unit="chain:full",
                         cause="transient"), actor=HUMAN)
        check("the first automatic unblock of a unit runs", a["status"] == "executed", a["reasons"])
        check("a second one of the same unit is refused",
              b["status"] == "refused" and "at most one" in b["reasons"][0], b["reasons"])
        check("another unit may still be unblocked", c["status"] == "executed")
        check("a person is not limited by the executor's count", d["status"] == "executed")
    finally:
        w.close()


def section_su_writer():
    print("SU ledger writer from report.gpu_hours")
    rep = json.loads((PILOT / "report.json").read_text())

    class Audited(dict):
        def __init__(self, d):
            super().__init__(d)
            self.read = set()

        def get(self, k, default=None):
            self.read.add(k)
            return super().get(k, default)

        def __getitem__(self, k):
            self.read.add(k)
            return super().__getitem__(k)

    w = World()
    try:
        au = Audited(rep)
        out = B.record_report_spend(au, "weedinc1", base_dir=w.ctx.su_base_dir)
        check("pilot_v1's report is recorded unit by unit",
              out["ok"] and len(out["recorded"]) == len(rep["gpu_hours"]), out)
        check("  its SU equal its GPU-hours at the V100 rate",
              abs(out["su"] - rep["gpu_hours_total"]) < 1e-6, (out["su"], rep["gpu_hours_total"]))
        check("  the writer read only exp, done, done_utc and gpu_hours",
              au.read <= set(B.REPORT_KEYS), sorted(au.read))
        again = B.record_report_spend(rep, "weedinc1", base_dir=w.ctx.su_base_dir)
        st = B.spent("weedinc1", base_dir=w.ctx.su_base_dir)
        check("recording it again updates, never double-bills",
              again["updated"] == len(rep["gpu_hours"]) and abs(st["su"] - out["su"]) < 1e-6
              and st["settled_exps"] == ["pilot_v1"], st)
        w2 = World()
        try:
            pert = copy.deepcopy(rep)
            for row in pert.get("final") or []:
                for exam in (row.get("exams") or {}):
                    if exam != "dev":
                        row["exams"][exam] = {"twelve": {"mean": 0.0}, "agnostic": {"mean": 0.0}}
            out2 = B.record_report_spend(pert, "weedinc1", base_dir=w2.ctx.su_base_dir)
            check("test / ood values changed: the recorded spend is identical",
                  out2["su"] == out["su"] and out2["recorded"] == out["recorded"])
        finally:
            w2.close()
        check("a report that is not done is refused",
              not B.record_report_spend(dict(rep, done=False), "weedinc1",
                                        base_dir=w.ctx.su_base_dir)["ok"])
        check("a report with no valid experiment name is refused",
              not B.record_report_spend(dict(rep, exp="../x"), "weedinc1",
                                        base_dir=w.ctx.su_base_dir)["ok"])
    finally:
        w.close()


def section_approvals_compat():
    print("approvals.py keeps its semantics")
    w = World()
    try:
        a = AP.propose("weed", "roboflow_generate_versions", {"n": 1}, "R3", "tier1:x", "why", T0,
                       root=w.dir)
        b = AP.propose("weed", "inc_build_pilot", {"exp": "p"}, "R3", AUTO, "why", T0 + 1,
                       root=w.dir, context={"lever": "L1"})
        raw = [x for x in AP.read("weed", root=w.dir)]
        check("a request filed without context has no context key",
              "context" not in raw[0] and raw[1].get("context") == {"lever": "L1"})
        check("propose refuses a context that is not an object",
              not AP.propose("weed", "x", {}, "R3", AUTO, "why", T0 + 2, root=w.dir,
                             context="L1")["ok"])
        for actor in ("tier0:g", "tier2:c", "round-scheduler", AUTO):
            check("decide still refuses %s" % actor,
                  not AP.decide("weed", a["item"]["id"], "approve", actor, "ok", T0 + 3,
                                root=w.dir)["ok"])
        with open(AP.path("weed", root=w.dir), "a") as fh:
            fh.write(json.dumps({"kind": "grant", "id": b["item"]["id"], "decision": "approved",
                                 "decided_by": "tier1:x", "ts": T0 + 4}) + "\n")
            fh.write(json.dumps({"kind": "executed", "id": a["item"]["id"], "phase": "started",
                                 "executed_by": AUTO, "ts": T0 + 5}) + "\n")
            fh.write(json.dumps({"kind": "grant", "id": a["item"]["id"], "decision": "approved",
                                 "decided_by": AUTO, "ts": T0 + 5, "grant": {}}) + "\n")
        st = w.items()
        check("a forged grant by a non-grantee is not applied",
              st[b["item"]["id"]]["status"] == "pending" and st[b["item"]["id"]]["attempts"])
        check("a grant line by the grantee on a non-INC item another tier filed is not applied",
              st[a["item"]["id"]]["status"] == "pending" and st[a["item"]["id"]]["attempts"]
              and AP.awaiting_execution("weed", root=w.dir) == [])
        check("an execution of an unapproved item is not applied",
              st[a["item"]["id"]].get("execution") is None
              and st[a["item"]["id"]]["execution_attempts"])
        AP.decide("weed", a["item"]["id"], "approve", HUMAN, "ok", T0 + 6, root=w.dir)
        check("awaiting_execution lists the approved, unexecuted item",
              [i["id"] for i in AP.awaiting_execution("weed", root=w.dir)] == [a["item"]["id"]])
    finally:
        w.close()


def main():
    section_policy_rows()
    section_authorize_matrix()
    section_envelope()
    section_envelope_scope()
    section_log_integrity()
    section_direct_once()
    section_cancel_sync()
    section_unblock_cause()
    section_resources_and_caps()
    section_retries_and_locks()
    section_replay_gate()
    section_budget()
    section_idempotency()
    section_r4()
    section_rendering()
    section_remote()
    section_unblock_once()
    section_su_writer()
    section_approvals_compat()
    print("\n%d failure(s)" % len(FAILURES))
    if FAILURES:
        for f in FAILURES:
            print("  - %s" % f)
    else:
        print("ALL PASS")
    return 1 if FAILURES else 0


def test_governance():
    """pytest entry point: the whole script, failing on any failed check."""
    del FAILURES[:]
    assert main() == 0, FAILURES


if __name__ == "__main__":
    sys.exit(main())
