"""
audit_registry_garbage.py — v3.0.71

Retroactively apply v3.0.68's strict_topic filter to slugs that were
harvested BEFORE strict mode existed. Drops registry entries (and on
--apply, the downloaded files too) for any slug with:

  - status == 'downloaded' AND
  - (labeled < threshold) OR (classes in (0, '?', None))

Default dry-run shows what would be dropped; --apply does the deed.

Wired as cluster_action 'audit_registry_garbage'. Click it once to clean
up the existing 7-slug noise (ibm_research/cif-dataset, random GH repos
with 0 labels). After this, v3.0.68 strict mode prevents new garbage
entering.

CLI:
    python -m weed_optimizer_framework.tools.audit_registry_garbage \\
        [--apply] [--labeled-min 100] [--keep-cwd12]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


REPO = Path(os.environ.get(
    "REPO_ROOT",
    "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark",
))
REGISTRY = REPO / "results" / "framework" / "dataset_registry.json"

# Slugs we never drop regardless of label count — framework baselines.
CWD12_BASELINES = {
    "cottonweed_holdout", "cottonweed_sp8", "cottonweeddet12",
}


def _slug_garbage_reason(slug: str, info: dict, labeled_min: int) -> str:
    """Return a short reason string if slug is garbage, else empty string.

    v3.0.71.1 (2026-06-01): fixed false-positive. Earlier version dropped
    on ANY non-numeric classes value ('?' or empty), which marked
    francesco__grass_weeds (2479 labeled bboxes!) as garbage because brain
    recorded its registry metadata as classes='?' despite the real schema.

    Correct semantics: drop ONLY if there's evidence of no usable labels.
    The disk check is authoritative — if there are >= labeled_min .txt
    YOLO label files on disk, the slug is fine regardless of registry
    metadata. This makes the audit safe even when brain's metadata is
    fuzzy.
    """
    if info.get("status") != "downloaded":
        return ""  # we only audit downloaded slugs

    labeled = info.get("labeled", info.get("local_labeled", 0)) or 0
    classes = info.get("classes")
    local_path = info.get("local_path", "")

    # First, authoritative disk check — count only .txt files that look like
    # YOLO labels (i.e., there's a matching image with the same stem).
    # v3.0.71.2: earlier version's rglob("*.txt") picked up README.txt and
    # other docs, falsely inflating the label count for ibm-cif (which has
    # 0 YOLO labels). Stem-match is the right discriminator.
    n_txt = 0
    if local_path and os.path.isdir(local_path):
        img_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        # Build set of image stems first (limit walk to avoid huge dirs)
        img_stems = set()
        for p in Path(local_path).rglob("*"):
            if p.suffix in img_exts:
                img_stems.add(p.stem)
                if len(img_stems) > 50000:
                    break
        # Count .txt files whose stem matches an image stem AND non-empty
        for p in Path(local_path).rglob("*.txt"):
            if p.stem in img_stems and p.stat().st_size > 0:
                n_txt += 1

    # Effective label count: max(registry-claimed labeled, disk yolo-label count)
    effective_labeled = max(int(labeled or 0), n_txt)

    if effective_labeled < labeled_min:
        # No usable labels anywhere → garbage.
        return (f"effective_labeled={effective_labeled} "
                f"(registry={labeled}, disk_txt={n_txt}) < {labeled_min}")

    # Has labels on disk: only secondary check is total absence of class info.
    if classes in (0, "0") and n_txt == 0:
        return f"classes=0 + no disk labels"

    return ""


def audit(labeled_min: int = 100, keep_cwd12: bool = True,
          apply: bool = False) -> dict:
    if not REGISTRY.exists():
        print(f"FATAL: {REGISTRY} missing", file=sys.stderr)
        sys.exit(2)
    with open(REGISTRY) as f:
        reg = json.load(f)
    ds = reg.get("datasets", {}) or {}

    before = len(ds)
    drops = []
    keeps = []
    for slug, info in sorted(ds.items()):
        if keep_cwd12 and slug in CWD12_BASELINES:
            keeps.append((slug, "cwd12 baseline (protected)"))
            continue
        reason = _slug_garbage_reason(slug, info, labeled_min)
        if reason:
            drops.append({
                "slug": slug,
                "reason": reason,
                "local_images": info.get("local_images", 0),
                "local_path": info.get("local_path", ""),
            })
        else:
            keeps.append((slug, "passed"))

    print(f"=== AUDIT (labeled_min={labeled_min}, keep_cwd12={keep_cwd12}, "
          f"apply={apply}) ===")
    print(f"  before: {before} slugs")
    print(f"  garbage candidates: {len(drops)}")
    for d in drops:
        print(f"    ✗ {d['slug']:<40} {d['reason']:<25} "
              f"({d['local_images']:,} imgs)")
    print(f"  keepers ({len(keeps)}):")
    for slug, why in keeps:
        local = ds[slug].get("local_images", 0)
        print(f"    ✓ {slug:<40} {why:<25} ({local:,} imgs)")

    if apply and drops:
        print(f"\n=== APPLY: removing {len(drops)} slugs ===")
        for d in drops:
            ds.pop(d["slug"], None)
            lp = d["local_path"]
            if lp and os.path.isdir(lp):
                try:
                    shutil.rmtree(lp)
                    print(f"  rmtree {lp}")
                except Exception as e:
                    print(f"  FAIL rmtree {lp}: {e}")
        # Refresh aggregate counter
        reg["datasets"] = ds
        reg["total_downloaded"] = sum(
            v.get("local_images", 0) for v in ds.values()
        )
        # Atomic save
        tmp = str(REGISTRY) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(reg, f, indent=2)
        os.replace(tmp, REGISTRY)
        print(f"  registry saved: {len(ds)} slugs remain, "
              f"{reg['total_downloaded']:,} imgs total")

    return {
        "ok": True,
        "before_slugs": before,
        "garbage_dropped": len(drops) if apply else 0,
        "garbage_candidates": [d["slug"] for d in drops],
        "after_slugs": (before - len(drops)) if apply else before,
        "applied": apply,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default: dry-run)")
    ap.add_argument("--labeled-min", type=int, default=100,
                    help="drop if labeled<N (default 100)")
    ap.add_argument("--no-keep-cwd12", action="store_true",
                    help="also drop cwd12 baselines if they fail")
    args = ap.parse_args()
    res = audit(labeled_min=args.labeled_min,
                keep_cwd12=not args.no_keep_cwd12,
                apply=args.apply)
    print(f"\n=== RESULT ===")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
