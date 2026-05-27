"""One-shot reclassifier for ALL classes already in the registry.

Usage on cluster (where Ollama / Gemma is reachable):
  python -m weed_optimizer_framework.tools.topic_backfill_all \\
      [--dry-run] [--model gemma3:27b] [--limit 50]

Without --dry-run, results write to class_topic_overrides.json.
With --limit, only the first N classes (alphabetical) are processed —
useful for a small test before committing to all 348.

Strategy:
  - Read all class_names from dataset_registry.json (union)
  - Skip classes already in overrides
  - For remaining: classify(use_llm=True) — hybrid keyword+LLM
  - Bulk-save results
  - Print summary by topic + by source

After run: /classes filter tabs reflect new groupings on next page load
(may need POST /api/refresh_registry to invalidate dashboard cache).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get(
    "REPO_ROOT",
    "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark",
))
sys.path.insert(0, str(REPO))

# Default override file = cluster path
os.environ.setdefault(
    "CLASS_TOPIC_OVERRIDES_FILE",
    str(REPO / "results" / "framework" / "class_topic_overrides.json"),
)

from weed_optimizer_framework.tools.topic_classifier import (
    classify_keyword, classify, classify_batch,
)
from weed_optimizer_framework.tools.class_topic_store import (
    load_overrides, save_overrides_bulk, VALID_TOPICS,
)

REGISTRY = REPO / "results" / "framework" / "dataset_registry.json"


def _collect_all_class_names() -> set:
    """Union of every class_names list across all slugs in registry."""
    if not REGISTRY.exists():
        raise SystemExit(f"FATAL: registry not found {REGISTRY}")
    with open(REGISTRY) as f:
        reg = json.load(f)
    out: set = set()
    for slug, info in (reg.get("datasets") or {}).items():
        for n in (info.get("class_names") or []):
            if isinstance(n, str) and n.strip():
                out.add(n.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be classified; no writes")
    ap.add_argument("--model", default="gemma3:27b",
                    help="Ollama model name for Brain LLM")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process first N alphabetical (test mode)")
    ap.add_argument("--no-llm", action="store_true",
                    help="keyword-only — useful when ollama down")
    args = ap.parse_args()

    print(f"==== topic backfill all — {time.strftime('%H:%M:%S')} ====")
    print(f"  registry: {REGISTRY}")
    print(f"  model:    {args.model}{'  (NOT USED, --no-llm)' if args.no_llm else ''}")

    all_classes = sorted(_collect_all_class_names())
    print(f"  total distinct class names: {len(all_classes)}")

    overrides = load_overrides()
    print(f"  already-classified (override exists): {len(overrides)}")

    todo = [c for c in all_classes if c not in overrides]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"  to classify this run:                 {len(todo)}")

    if not todo:
        print("  nothing to do — every class already has an override.")
        return

    print()
    if args.dry_run:
        # Preview: what keyword says vs nothing yet
        print("  DRY-RUN preview (keyword baseline only):")
        kw_counts = {}
        for c in todo:
            t, conf = classify_keyword(c)
            kw_counts[t] = kw_counts.get(t, 0) + 1
        for t in VALID_TOPICS:
            n = kw_counts.get(t, 0)
            if n:
                print(f"    keyword would assign {t:8s} → {n}")
        n_other = kw_counts.get("other", 0)
        print(f"    {'⚠ ' if n_other > 50 else ''}{n_other} would hit 'other' — "
              f"these are where the LLM helps most.")
        return

    # Live run
    start = time.time()
    results = classify_batch(
        todo, use_llm=not args.no_llm, model=args.model, persist=True,
    )
    elapsed = time.time() - start

    # Summary
    by_source = {}
    by_topic = {}
    disagreements = []
    for r in results:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        t = r.get("topic", "?")
        by_topic[t] = by_topic.get(t, 0) + 1
        if (r.get("keyword_topic") and r.get("llm_topic") and
                r["keyword_topic"] != r["llm_topic"]):
            disagreements.append(r)

    print()
    print(f"  done in {elapsed:.1f}s ({elapsed/len(todo):.2f}s/class)")
    print(f"  by source: {by_source}")
    print(f"  by topic:  {by_topic}")
    if disagreements:
        print(f"\n  ⚠ {len(disagreements)} kw↔LLM disagreements (using LLM):")
        for r in disagreements[:20]:
            print(f"    {r['cls']:30s} kw={r['keyword_topic']:8s} → llm={r['llm_topic']}")
        if len(disagreements) > 20:
            print(f"    ... +{len(disagreements) - 20} more")

    print()
    print(f"  override file now has {len(load_overrides())} entries.")
    print(f"  reload /classes to see updated topic groupings.")


if __name__ == "__main__":
    main()
