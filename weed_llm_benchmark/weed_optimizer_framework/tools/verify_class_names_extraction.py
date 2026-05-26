"""v3.0.42.5 verification: probe known-good HF datasets, see if our new
extraction logic correctly pulls ClassLabel.names from features.

Run on cluster (where HF cache is populated):
  python -m weed_optimizer_framework.tools.verify_class_names_extraction

For each candidate dataset we print:
  - features tree summary
  - extracted class names (the v3.0.42.5 fix)
  - whether the result is what we'd expect (sanity check)
"""
from __future__ import annotations
import os
import sys

# Ensure imports resolve from cluster repo root
sys.path.insert(0, os.environ.get(
    "REPO_ROOT",
    "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark",
))

from weed_optimizer_framework.tools.dataset_discovery import DatasetDiscovery


# (hf_id, config_to_try, expected_substring_in_names_or_None)
# Small, reliably-available HF object-detection datasets known to expose
# ClassLabel.names in features. Used for online sanity check.
CANDIDATES = [
    # CPPE-5: medical PPE detection, 5 classes, ~1K imgs, very small download
    ("cppe-5", None, "Coverall"),
    # Keremberke garbage detection — popular small detection set
    ("keremberke/garbage-object-detection", "mini", None),
]


def _local_features_test() -> bool:
    """Offline test: build mock HF Features with a nested ClassLabel.
    Verifies the recursive walker without depending on network."""
    try:
        from datasets import Features, ClassLabel, Sequence, Image, Value
    except Exception as e:
        print(f"  SKIP local test: datasets import failed: {e}")
        return False
    print("\n========== LOCAL: mock HF Features (no network) ==========")
    fake = Features({
        'image': Image(),
        'objects': Sequence({
            'bbox': Sequence(Value('float32'), length=4),
            'category': ClassLabel(names=['weed', 'crop', 'background']),
        }),
    })
    names = DatasetDiscovery._extract_class_names_from_hf_features(fake)
    ok = names == ['weed', 'crop', 'background']
    print(f"  features keys: {list(fake.keys())}")
    print(f"  extracted: {names}")
    print(f"  expected:  ['weed', 'crop', 'background']")
    print(f"  result: {'✓ PASS' if ok else '✗ FAIL'}")

    # Test 2: flat ClassLabel at top level
    fake2 = Features({
        'image': Image(),
        'label': ClassLabel(names=['cls_a', 'cls_b']),
    })
    names2 = DatasetDiscovery._extract_class_names_from_hf_features(fake2)
    ok2 = names2 == ['cls_a', 'cls_b']
    print(f"  flat-ClassLabel: extracted={names2}  → {'✓' if ok2 else '✗'}")

    # Test 3: no ClassLabel at all
    fake3 = Features({'image': Image(), 'caption': Value('string')})
    names3 = DatasetDiscovery._extract_class_names_from_hf_features(fake3)
    ok3 = names3 == []
    print(f"  no-ClassLabel: extracted={names3}  → {'✓' if ok3 else '✗'}")

    return ok and ok2 and ok3


def main():
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"FATAL: datasets not installed: {e}")
        sys.exit(2)

    # Always run the offline test first — it doesn't depend on HF availability.
    local_ok = _local_features_test()

    pass_n = 1 if local_ok else 0
    fail_n = 0 if local_ok else 1
    for hf_id, cfg, expect_sub in CANDIDATES:
        print(f"\n========== {hf_id} (cfg={cfg}) ==========")
        try:
            ds = (load_dataset(hf_id, cfg, split="train", streaming=True)
                  if cfg else load_dataset(hf_id, split="train", streaming=True))
        except Exception as e:
            print(f"  LOAD FAIL: {str(e)[:200]}")
            fail_n += 1
            continue
        feats = getattr(ds, "features", None)
        if feats is None:
            print("  no features attribute")
            fail_n += 1
            continue
        print(f"  feature keys: {list(feats.keys()) if hasattr(feats, 'keys') else '?'}")
        # Pretty-print top-level feature types
        if hasattr(feats, "items"):
            for k, v in feats.items():
                print(f"    {k}: {type(v).__name__}")

        names = DatasetDiscovery._extract_class_names_from_hf_features(feats)
        if names:
            print(f"  ✓ EXTRACTED ({len(names)}): {names[:10]}"
                  f"{'  ...' if len(names) > 10 else ''}")
            if expect_sub:
                hit = any(expect_sub.lower() in n.lower() for n in names)
                print(f"  expected substring '{expect_sub}' in names: {hit}")
                if hit:
                    pass_n += 1
                else:
                    fail_n += 1
            else:
                pass_n += 1
        else:
            print("  ✗ NO names extracted")
            fail_n += 1

    print(f"\n==== summary: pass={pass_n}  fail={fail_n} ====")
    sys.exit(0 if pass_n > 0 else 1)


if __name__ == "__main__":
    main()
