#!/usr/bin/env python3
"""Phase-1 domain-config unit tests (no Mongo / cluster / Roboflow needed).

Exercises the pure logic that routes weed hardcodes through the per-domain
config layer:
  * db.DEFAULT_DOMAIN_CONFIG shape + db._deep_merge
  * DatasetDiscovery._resolve_domain_config honours an EXPLICIT accept_vocab
    (config wins) and otherwise DERIVES vocab from taxonomy/queries.

Run:  python -m pytest tests/test_domain_config.py     (or) python tests/test_domain_config.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools import db  # noqa: E402
from weed_optimizer_framework.tools import dataset_discovery as dd  # noqa: E402

_fails = []


def ck(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        _fails.append(name)


# ---- db.DEFAULT_DOMAIN_CONFIG + _deep_merge -------------------------------
D = db.DEFAULT_DOMAIN_CONFIG
ck("default config has thresholds", isinstance(D.get("thresholds"), dict))
for k in ("dino_threshold", "imbalance_high", "imbalance_med", "dup_frac",
          "tiny_px", "min_per_class", "small_dataset"):
    ck(f"default threshold '{k}' present", k in D["thresholds"])

merged = db._deep_merge({"a": {"x": 1, "y": 2}, "b": 5},
                        {"a": {"y": 9, "z": 3}})
ck("_deep_merge keeps un-overridden key", merged["a"]["x"] == 1)
ck("_deep_merge overrides nested key", merged["a"]["y"] == 9)
ck("_deep_merge adds new nested key", merged["a"]["z"] == 3)
ck("_deep_merge does not mutate base default", D["thresholds"]["dino_threshold"] == 0.45)


# ---- _resolve_domain_config: explicit accept_vocab wins -------------------
brain = object.__new__(dd.DatasetDiscovery)  # skip __init__ (no FS/registry)

_orig_get_domain = db.get_domain
try:
    db.get_domain = lambda dom: {
        "display_name": "Coral",
        "harvest_queries": ["coral reef detection dataset"],
        "taxonomy": ["staghorn coral"],
        "accept_vocab": ["coral", "reef", "polyp"],
    }
    cfg = brain._resolve_domain_config("coral")
    ck("explicit accept_vocab used verbatim",
       set(cfg["accept"]) == {"coral", "reef", "polyp"})
    ck("queries flow through", cfg["queries"] == ["coral reef detection dataset"])

    # No explicit accept_vocab → derive from taxonomy/queries/name.
    db.get_domain = lambda dom: {
        "display_name": "Coral",
        "harvest_queries": ["coral reef detection dataset"],
        "taxonomy": ["staghorn coral"],
    }
    cfg2 = brain._resolve_domain_config("coral")
    ck("derived vocab includes taxonomy word", "staghorn" in cfg2["accept"])
    ck("derived vocab drops stopword 'dataset'", "dataset" not in cfg2["accept"])
finally:
    db.get_domain = _orig_get_domain


# ---- model_router.resolve (Phase 2) --------------------------------------
from weed_optimizer_framework.tools import model_router as mr  # noqa: E402

OLLAMA_UP = {"ollama": {"configured": True}}
OLLAMA_DOWN = {"ollama": {"configured": False}}

# lab role, ollama reachable → the small default answers
r = mr.resolve("analysis_summary", provider_status=OLLAMA_UP)
ck("lab role resolves to small default", r["model"] == mr.LAB_SMALL and r["place"] == "lab")
ck("lab role source=default", r["source"] == "default")
ck("lab role reachable when ollama up", r["reachable"] is True)

# lab role, ollama DOWN → still returns a model but honestly flags unreachable
r = mr.resolve("analysis_summary", provider_status=OLLAMA_DOWN)
ck("lab role unreachable flagged", r["reachable"] is False and r["source"] == "unreachable_default")

# per-domain override wins
r = mr.resolve("analysis_summary",
               domain_config={"model_routing": {"analysis_summary": "ollama:qwen2.5:3b"}},
               provider_status=OLLAMA_UP)
ck("domain override taken", r["source"] == "domain")

# global role config used when no domain override
r = mr.resolve("interactive_plan", global_roles={"interactive_plan": "ollama:qwen2.5:3b"},
               provider_status=OLLAMA_UP)
ck("global role taken", r["source"] == "global")

# cluster role: reachable is contextual (True) even with empty status; place=cluster
r = mr.resolve("harvest_brain", provider_status={})
ck("cluster role place", r["place"] == "cluster")
ck("cluster role reachable in-context", r["reachable"] is True)
ck("cluster role default model", r["model"] == "ollama:gemma4")

# hard_reasoning is async + rare + cluster
r = mr.resolve("hard_reasoning", provider_status={})
ck("hard_reasoning is async", r["is_async"] is True)
ck("hard_reasoning is rare", r["rare"] is True)

# unknown role → ok False, safe lab fallback
r = mr.resolve("does_not_exist")
ck("unknown role ok=False", r["ok"] is False and r["model"] == mr.LAB_SMALL)

# vllm on a lab role is NOT reachable from the lab (cluster-only endpoint)
ck("vllm not reachable on lab", mr._reachable("vllm:glm-4.7-flash", "lab", OLLAMA_UP) is False)
ck("role_table lists all roles", len(mr.role_table()) == len(mr.ROLES))


# ---- round provenance pure helpers (Phase 4) -----------------------------
ck("ROUND_STEPS is the 5-step loop",
   db.ROUND_STEPS == ["collect", "filter", "label", "train", "eval"])
ck("_round_id formats domain#n", db._round_id("coral", 3) == "coral#3")

e = db._round_step_entry("running", detail={"x": 1}, job="j123", actor="alice", now="T")
ck("step entry keeps valid status", e["status"] == "running")
ck("step entry records actor+at+job", e["actor"] == "alice" and e["at"] == "T" and e["job"] == "j123")
ck("step entry keeps detail", e["detail"] == {"x": 1})

e2 = db._round_step_entry("bogus", now="T")
ck("invalid status coerced to pending", e2["status"] == "pending")
ck("step entry omits job when none", "job" not in e2)

# record_round_step rejects an unknown step (before any Mongo call)
ck("record_round_step rejects bad step", db.record_round_step("coral", "notastep", "done") is None)


# ---- dataset_eda.analyze_nonimage (Phase 5 extraction, pure) -------------
import tempfile as _tf  # noqa: E402
import os as _os  # noqa: E402
from pathlib import Path as _P  # noqa: E402
from weed_optimizer_framework.tools.dataset_eda import analyze_nonimage  # noqa: E402

_d = _tf.mkdtemp()
with open(_os.path.join(_d, "s.csv"), "w") as _f:
    _f.write("t,ax,label\n0,0.1,walk\n1,0.2,walk\n2,0.05,idle\n")
_r = analyze_nonimage(_P(_d))
ck("analyze_nonimage finds sensor bucket", "sensor" in _r)
ck("analyze_nonimage detects label column", (_r.get("sensor") or {}).get("label_column") == "label")
ck("analyze_nonimage counts 2 classes", (_r.get("sensor") or {}).get("n_classes") == 2)


# ---- sealed lever menu (v3.28.0) -----------------------------------------
# The menu is pre-registered evidence, not a convenience list: an experiment
# whose lever was picked after the result is not an experiment, and a lever
# reported without its control reports the campaign's drift instead of the
# lever's effect. These checks are what keep a later edit from quietly
# loosening either property.
_LM = D.get("lever_menu") or []
ck("lever_menu is present and non-empty", len(_LM) > 0)
ck("every lever has a stable id", all(str(l.get("id") or "").strip() for l in _LM))
ck("lever ids are unique", len({l.get("id") for l in _LM}) == len(_LM))
ck("every lever names a control", all(str(l.get("control") or "").strip() for l in _LM))
ck("every lever states a reason", all(len(str(l.get("reason") or "")) > 40 for l in _LM))
ck("every lever declares options", all(l.get("options") not in (None, [], {}) for l in _LM))
ck("every lever names the step it applies to",
   all(l.get("applies_to") in ("collect", "filter", "train") for l in _LM))
ck("every lever carries a risk tier",
   all(str(l.get("risk") or "") in ("R1", "R2", "R3", "R4") for l in _LM))
ck("no lever is R4 (an experiment must never be irreversible)",
   all(l.get("risk") != "R4" for l in _LM))
ck("every lever carries an SU estimate",
   all(isinstance(l.get("est_su"), (int, float)) and l["est_su"] >= 0 for l in _LM))
ck("the recipe comparison is on the menu with cwd12_core as its control",
   any(l["id"] == "core_recipe" and "cwd12_core" in str(l.get("control")) for l in _LM))
ck("every recipe named by the core_recipe lever has a sealed noise floor",
   all(o.split("+")[0] in D["noise_floor"]
       for l in _LM if l["id"] == "core_recipe" for o in l["options"]))


if _fails:
    print(f"\nFAILED: {len(_fails)} -> {_fails}")
    sys.exit(1)
print("\nALL PASS")
