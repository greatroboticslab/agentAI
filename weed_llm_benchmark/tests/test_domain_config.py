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


if _fails:
    print(f"\nFAILED: {len(_fails)} -> {_fails}")
    sys.exit(1)
print("\nALL PASS")
