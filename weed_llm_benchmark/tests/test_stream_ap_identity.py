#!/usr/bin/env python3
"""Experiment mode is byte-identical before and after stream mode
(docs/CONTINUOUS_LOOP.md 9, group F acceptance: "an experiment-mode
campaign's digest and argv are byte-identical before and after").

The package as committed at HEAD (git archive, read only) and the working
tree each run the same experiment-mode computations in a subprocess, on the
pinned replay fixtures, and print one JSON record:

  * the rules version (diagnose.rules_version: diagnose.py, thresholds.json,
    levers.json, levers.py are untouched by stream mode). A later funnel
    change moves it until committed (levers.json, levers.py): it may differ
    only when those two files alone changed and levers.json's experiment-mode
    part (every non-funnel lever row, the cards, operations, refusals, _meta
    and the protocol constants but funnel_*) is identical; the rest of the
    record is compared byte for byte either way;
  * for pilot_v1, pilot_v2, pilot_v3 and realloop_v1 (the whole tree as each
    replay case loads it): every diagnosis (id, fired, severity, summary,
    levers, cites, detail) and levers.propose's proposals (argv, params,
    policy action, risk, trigger, cites, est_gpu_hours), cards, operations and
    deferred items;
  * the brain digest of pilot_v3 and of realloop_v1 (brain_plan.build_digest,
    sha256 and bytes, at a fixed clock);
  * the evidence's canonical sha256 of every experiment;
  * executor.render of every v1 action's argv and remote line, and a budget
    state of an experiment campaign (budget.state) from a fixed execution log;
  * the policy rows of every v1 action (policy.describe);
  * the approvals' envelope scope of a v1 build.

The two records must be equal byte for byte. Needs git and a HEAD that holds
the autopilot; a checkout without them skips with the reason printed.

Run:  python3 tests/test_stream_ap_identity.py
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
FAILURES, SKIPS = [], []

PROBE = r'''
import json, sys, hashlib, os
root = sys.argv[1]; fix = sys.argv[2]
sys.path.insert(0, root)
os.environ["INC_DIR"] = os.path.join(sys.argv[3], "inc")
from weed_optimizer_framework.tools.inc_autopilot import evidence as E, diagnose as DG, levers as LV
from weed_optimizer_framework.tools.inc_autopilot import brain_plan as BP, executor as X, budget as B, model as M
from weed_optimizer_framework.tools.brain import policy as POL, approvals as AP
out = {"rules_version": DG.rules_version(), "rules_files": DG.rules_digest()["files"]}
_menu = json.loads(open(LV.MENU_FILE).read())
_fun = set(LV.FUNNEL_LEVERS)
out["levers_json_experiment"] = hashlib.sha256(json.dumps(
    {"levers": {k: v for k, v in _menu["levers"].items() if k not in _fun}, "cards": _menu.get("cards"),
     "operations": _menu.get("operations"), "refusals": _menu.get("refusals"), "_meta": _menu.get("_meta"),
     "protocol": {k: v for k, v in _menu["protocol"].items() if not k.startswith("funnel_")}},
    sort_keys=True).encode()).hexdigest()
def diag_rec(ds):
    return [{k: d.get(k) for k in ("id", "fired", "severity", "summary", "levers", "cites", "detail", "exp")} for d in ds]
trees = {"pilot_v1": ["pilot_v1", "b0_v1"], "pilot_v2": ["pilot_v1", "pilot_v2", "b0_v1", "base_b_v1"],
         "pilot_v3": None}
for exp, exps in sorted(trees.items()):
    ev = E.load_dir(fix, exp, exps=exps)
    ds = DG.detect(ev)
    prop = LV.propose(ds, ev)
    out[exp] = {"canonical": hashlib.sha256(ev.canonical()).hexdigest(), "diagnoses": diag_rec(ds),
                "proposals": [{k: p.get(k) for k in ("lever", "argv", "params", "policy_action", "risk", "trigger",
                                                     "cites", "est_gpu_hours", "child_exp", "parent_exp")}
                              for p in prop.get("proposals") or []],
                "cards": [{k: c.get(k) for k in ("lever", "title", "trigger")} for c in prop.get("cards") or []],
                "operations": prop.get("operations"), "deferred": prop.get("deferred")}
    if exp == "pilot_v3":
        dg = BP.build_digest(ev, exp, ds, BP.load_menu(), campaign="c", n=1, track={}, lineage=[],
                             budget={"envelope_su": 300.0, "spent_su": 0.0, "committed_su": 0.0, "remaining_su": 300.0},
                             residuals=[], deterministic=prop.get("proposals") or [], corpus=None,
                             parents=[e for e in ev.exps() if e != exp][-3:], created_utc="2026-09-28T00:00:00Z")
        out[exp]["digest_sha256"] = dg.get("sha256")
        out[exp]["digest_bytes"] = hashlib.sha256(json.dumps(dg, sort_keys=True, default=str).encode()).hexdigest()
rv = os.path.join(fix, "funnel")
ev = E.load_dir(rv, "realloop_v1", exps=["realloop_v1"])
ds = DG.detect(ev)
prop = LV.propose(ds, ev)
dg = BP.build_digest(ev, "realloop_v1", ds, BP.load_menu(), campaign="c", n=1, track={}, lineage=[], budget=None,
                     residuals=[], deterministic=prop.get("proposals") or [], corpus=None, parents=[],
                     created_utc="2026-09-28T00:00:00Z")
out["realloop_v1"] = {"canonical": hashlib.sha256(ev.canonical()).hexdigest(), "diagnoses": diag_rec(ds),
                      "proposals": [p.get("argv") for p in prop.get("proposals") or []], "digest_sha256": dg.get("sha256")}
acts = {"inc_build_pilot": {"exp": "pilot_v9", "replay_mode": "full", "gate_flips_mode": "net"},
        "inc_build_baseline": {"exp": "b9", "manifest": "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc/step1/base_B.jsonl"},
        "inc_build_realloop": {"exp": "r9", "replay_mode": "full", "recipes": "full", "increment_sources": "evidence",
                               "size": 287, "n_verified": 4, "gate_flips_mode": "net"},
        "inc_label_audit": {"trusted": "/x/a.jsonl", "audit": "I1=/x/b.jsonl,I2=/x/c.jsonl", "out": "/x/o.json"},
        "inc_unblock_transient": {"exp": "e", "unit": "base", "cause": "oom"},
        "inc_snapshot": {"exp": "e", "ledger_from": 3}, "inc_cancel_exp": {"exp": "e"}, "inc_sync_outer": {},
        "inc_relevance_build": {"sample": 300}}
out["render"] = {a: X.render(a, p, {"parent_exp": "p", "trigger": "D1", "approval_id": "ap-1", "decided_by": "human:x"})
                 for a, p in sorted(acts.items())}
out["policy"] = {a: POL.describe(a) for a in sorted(acts) + ["round_collect", "round_filter", "round_train",
                                                             "inc_funnel_audit", "inc_funnel_fetch", "inc_plan_submit"]}
out["envelope_scope"] = AP.envelope_scope({"status": "pending", "risk": "R3", "action": "inc_build_pilot",
                                           "requested_by": "round-scheduler:inc-autopilot"})
execs = [{"campaign": "weedinc", "action": "inc_build_pilot", "params": {"exp": "pilot_v9"}, "charged": True,
          "est_su": 40.0, "epoch": 1790000000.0, "ts": "2026-09-21T00:00:00Z", "status": "executed", "run_id": "r1"},
         {"campaign": "weedinc", "action": "inc_label_audit", "params": {}, "charged": True, "est_su": 2.0,
          "epoch": 1790000100.0, "ts": "2026-09-21T00:01:40Z", "status": "executed", "run_id": "r2", "job_ids": ["77"]}]
out["budget"] = B.state({"name": "weedinc", "autonomy": "envelope", "envelope_su": 300, "daily_cap_su": 120}, execs,
                        None, 1790000200.0, base_dir=os.path.join(sys.argv[3], "brain"))
out["fits"] = B.fits(out["budget"], 250.0, need_daily=True)
out["replay_required"] = list(X.REPLAY_REQUIRED)
out["envelope_levers"] = {k: list(v) for k, v in X.ENVELOPE_LEVERS.items()}
print("IDENTITY " + json.dumps(out, sort_keys=True, default=str))
'''


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:2000]))
        FAILURES.append(name)


def probe(pkg_root, fix, work):
    p = subprocess.run([sys.executable, "-c", PROBE, str(pkg_root), str(fix), str(work)], capture_output=True,
                       text=True, timeout=900, cwd=str(pkg_root))
    for line in (p.stdout or "").splitlines():
        if line.startswith("IDENTITY "):
            return json.loads(line[len("IDENTITY "):]), ""
    return None, (p.stdout or "")[-1500:] + (p.stderr or "")[-1500:]


def diff(a, b, path="", out=None):
    out = [] if out is None else out
    if type(a) != type(b):
        out.append((path, str(a)[:200], str(b)[:200]))
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((path + "/" + k, str(a.get(k))[:200], str(b.get(k))[:200]))
            else:
                diff(a[k], b[k], path + "/" + k, out)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append((path, "len %d" % len(a), "len %d" % len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, "%s/%d" % (path, i), out)
    elif a != b:
        out.append((path, str(a)[:200], str(b)[:200]))
    return out


def main():
    print("experiment mode: byte-identical to HEAD")
    top = ROOT.parent
    try:
        r = subprocess.run(["git", "-C", str(top), "rev-parse", "--verify", "HEAD"], capture_output=True, text=True,
                           timeout=60)
    except OSError as e:
        r = None
        why = "git is not available (%s)" % e
    if r is None or r.returncode != 0:
        print("  skip: %s" % ("no git HEAD" if r is not None else why))
        SKIPS.append("identity (no git HEAD)")
        print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), ", ".join(SKIPS)))
        return 0
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="stream_identity_"))
    try:
        ls = subprocess.run(["git", "-C", str(top), "ls-tree", "--name-only", "HEAD", "weed_llm_benchmark/"],
                            capture_output=True, text=True, timeout=60)
        scripts = [x for x in ls.stdout.split() if x.endswith(".sh")]
        arch = subprocess.run(["git", "-C", str(top), "archive", "--format=tar", "HEAD",
                               "weed_llm_benchmark/weed_optimizer_framework"] + scripts, capture_output=True,
                              timeout=300)
        if arch.returncode != 0:
            print("  skip: git archive failed: %s" % arch.stderr[-300:])
            SKIPS.append("identity (git archive)")
            print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), ", ".join(SKIPS)))
            return 0
        subprocess.run(["tar", "-x", "-C", str(tmp)], input=arch.stdout, check=True, timeout=300)
        head_root = tmp / "weed_llm_benchmark"
        fix = ROOT / "tests" / "fixtures" / "inc_replay"
        wa, wb = tmp / "work_head", tmp / "work_now"
        wa.mkdir()
        wb.mkdir()
        a, ea = probe(head_root, fix, wa)
        b, eb = probe(ROOT, fix, wb)
        check("HEAD's package ran the experiment-mode probe", a is not None, ea)
        check("the working tree's package ran it", b is not None, eb)
        if a is None or b is None:
            print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), ", ".join(SKIPS) or "none"))
            return 1
        a_s, b_s = json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True)
        a_s = a_s.replace(str(wa), "<W>").replace(str(head_root), "<R>")
        b_s = b_s.replace(str(wb), "<W>").replace(str(ROOT), "<R>")
        d = diff(json.loads(a_s), json.loads(b_s))
        # The rules version hashes diagnose.py, thresholds.json, levers.json and
        # levers.py whole, so a funnel change (the funnel levers' rows and code,
        # the funnel_* protocol constants) moves it until it is committed. It
        # may move only that way: the files that changed are levers.json and
        # levers.py, and levers.json's experiment-mode rows, cards, operations,
        # refusals and protocol are the same; every record below is still
        # compared byte for byte.
        ra, rb = a.get("rules_files") or {}, b.get("rules_files") or {}
        changed = sorted(k for k in set(ra) | set(rb) if ra.get(k) != rb.get(k))
        if a.get("rules_version") == b.get("rules_version"):
            check("the rules version is unchanged (%s)" % a.get("rules_version"), True)
        else:
            check("the rules version moved (%s -> %s) for a funnel change only: %s changed; levers.json's "
                  "experiment-mode rows, cards, operations, refusals and protocol are identical"
                  % (a.get("rules_version"), b.get("rules_version"), ", ".join(changed)),
                  changed and set(changed) <= {"levers.json", "levers.py"}
                  and a.get("levers_json_experiment") == b.get("levers_json_experiment"),
                  (changed, a.get("levers_json_experiment"), b.get("levers_json_experiment")))
        rules_keys = ("rules_version", "rules_files", "levers_json_experiment")
        a_s = json.dumps({k: v for k, v in json.loads(a_s).items() if k not in rules_keys}, sort_keys=True)
        b_s = json.dumps({k: v for k, v in json.loads(b_s).items() if k not in rules_keys}, sort_keys=True)
        d = [x for x in d if not x[0].startswith(tuple("/" + k for k in rules_keys))]
        for key in ("pilot_v1", "pilot_v2", "pilot_v3", "realloop_v1"):
            check("%s: diagnoses, proposals, argv, cards and evidence identical" % key,
                  not [x for x in d if x[0].startswith("/" + key) and "digest" not in x[0]],
                  [x for x in d if x[0].startswith("/" + key)][:5])
        check("the brain digests of pilot_v3 and realloop_v1 are identical",
              a["pilot_v3"]["digest_sha256"] == b["pilot_v3"]["digest_sha256"]
              and a["pilot_v3"]["digest_bytes"] == b["pilot_v3"]["digest_bytes"]
              and a["realloop_v1"]["digest_sha256"] == b["realloop_v1"]["digest_sha256"],
              [x for x in d if "digest" in x[0]][:5])
        for key in ("render", "policy", "envelope_scope", "budget", "fits", "replay_required", "envelope_levers"):
            check("%s identical" % key, not [x for x in d if x[0].startswith("/" + key)],
                  [x for x in d if x[0].startswith("/" + key)][:5])
        check("the whole record (the rules version aside, checked above) is byte-identical", a_s == b_s, d[:8])
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)
    print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), ", ".join(SKIPS) or "none"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
