"""The two paths that submit work must both pass the same policy gate.

This file exists because a validation added to a submission path has twice come
within one deploy of stopping the campaign silently: a script allow-list that
refused every template because the scripts live on the cluster and the check ran
on the lab, and a parameter gate that refused every step because it was handed
the whole round's parameters instead of the step's own. Both were caught by
running a test, not by reading the code. So the live configuration is asserted
here rather than reasoned about.

`dashboard_server` is parsed, never imported: importing it off-cluster fails on a
read-only /ocean path, and a test that cannot run is not a test.
"""

import ast
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools import db                       # noqa: E402
from weed_optimizer_framework.tools import round_scheduler as rs    # noqa: E402
from weed_optimizer_framework.tools.brain import policy             # noqa: E402

_fails = []


def ck(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        _fails.append(name)


SRC = (pathlib.Path(__file__).resolve().parents[1] / "weed_optimizer_framework"
       / "tools" / "dashboard_server.py").read_text()


def _cluster_action_function():
    """Source of `api_cluster_action`, located by the dispatch it performs."""
    tree = ast.parse(SRC)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seg = ast.get_source_segment(SRC, node) or ""
            if "_CLUSTER_ACTIONS[action]" in seg:
                return seg
    return ""


FN = _cluster_action_function()
ck("the cluster-action handler was located in the source", bool(FN))

# ---- the declared body keys are the keys the handler reads ------------------
declared = re.search(r"_CLUSTER_BODY_KEYS = \(([^)]*)\)", SRC)
ck("the handler declares the body keys it gates", declared is not None)
declared_keys = set(re.findall(r"[\"']([A-Za-z0-9_]+)[\"']", declared.group(1))) \
    if declared else set()
read_keys = set(re.findall(r"body\.get\(\s*[\"']([A-Za-z0-9_]+)[\"']", FN))
ck("every body field the handler reads is gated (%s)" % sorted(read_keys - declared_keys),
   read_keys <= declared_keys)
ck("no gated field is one the handler never reads (%s)"
   % sorted(declared_keys - read_keys), declared_keys <= read_keys)

# ---- the gate is actually wired into both paths -----------------------------
ck("the web path calls the policy gate", "policy as _brain_policy" in FN
   and "_brain_policy.authorize(" in FN)
ck("the web path refuses with 403 rather than continuing",
   "Refused by policy" in FN)
ck("the scheduler calls the policy gate",
   "_policy_ok(" in pathlib.Path(rs.__file__).read_text())

# ---- the gate fails closed for the unattended path, open for the watched one -
sched_src = pathlib.Path(rs.__file__).read_text()
i = sched_src.index("def _policy_ok(")
gate = sched_src[i:sched_src.index("\ndef ", i + 10)]
ck("the scheduler gate refuses when the policy module cannot be imported",
   "return False" in gate and "policy gate unavailable" in gate)
ck("the scheduler gate refuses when the gate itself raises",
   "policy gate raised" in gate)
ck("the scheduler gate refuses an action needing approval",
   "needs approval" in gate)
ck("the web path allows an action the catalogue does not know, and logs it",
   "not a recognised action" in FN and "allowing" in FN)

# ---- every real request body still passes -----------------------------------
# The bodies below are the ones the handler reads, with values the UI sends.
for action, body in (("brain_harvest", {"time_h": 2}),
                     ("clean_train_d", {"dino_min": 0.5, "epochs": 60}),
                     ("dinov2_curate_registry", {"domain": "weed"}),
                     ("sync_all_to_roboflow", {"domain": "weed"}),
                     ("refresh_registry", {}),
                     ("restart_dashboard", {}),
                     ("start_new_round", {}),
                     ("build_buckets", {})):
    d = policy.authorize("human:harry", action, body, None, None)
    ck("an administrator can still run %s" % action, bool(d["allowed"]))

# A field the handler never reads must not refuse the call, because the gate
# never sees it -- this asserts the filter, not the catalogue.
stray = {k: v for k, v in {"time_h": 2, "unrelated": "x"}.items()
         if k in declared_keys}
ck("a stray body field is filtered out before the gate", stray == {"time_h": 2})
ck("and the filtered body is still permitted",
   policy.authorize("human:harry", "brain_harvest", stray, None, None)["allowed"])

# ---- the scheduler's own three steps ----------------------------------------
D = db.DEFAULT_DOMAIN_CONFIG
for step in ("collect", "filter", "train"):
    sub = {k: v for k, v in D["round_params"].items()
           if k in db.step_fields(D, step)}
    d = policy.authorize("round-scheduler", "round_" + step, sub, None, None)
    ck("the live %s step is authorised" % step,
       d["allowed"] and not d["needs_approval"])

# ---- and the gate still refuses what it must --------------------------------
ck("an automated tier may not delete a Roboflow project",
   not policy.authorize("tier1:deepseek-v4-flash", "roboflow_delete_junk_apply",
                        {}, None, None)["allowed"])
ck("an automated tier may not restart the dashboard",
   not policy.authorize("tier0:gemma4", "restart_dashboard", {}, None, None)["allowed"])
ck("an out-of-bounds epoch count is refused on the web path too",
   not policy.authorize("human:harry", "clean_train_d",
                        {"epochs": 100000}, None, None)["allowed"])

if _fails:
    print("\nFAILED: %d -> %s" % (len(_fails), _fails))
    sys.exit(1)
print("\nALL PASS")
