"""
dataset_integrity_audit.py — Master-plan P1: data integrity & deep audit.

  --audit   READ-ONLY:
            1) HOLDOUT PROTECTION: does the cwd12 test/valid stem source exist on
               disk? how many stems? (mega_trainer's stem-level leak defence is
               only as good as this set — 0 stems = NO protection = leak risk.)
            2) PER-CLASS trainable image counts via bounded label scan (per CWD12
               species, across non-holdout bbox slugs).

  --mark-overall-test
            Flag the overall hand-verified NEVER-TRAIN test set in Mongo
            (is_overall_test + cwd12_eval_holdout, provenance=gold) and VERIFY a
            train-eligible query excludes them.

Run on the cluster:
    python -m weed_optimizer_framework.tools.dataset_integrity_audit --audit
    python -m weed_optimizer_framework.tools.dataset_integrity_audit --mark-overall-test
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

from .registry_lock import safe_read_json

REPO = Path(os.environ.get(
    "REPO_ROOT",
    "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark",
)).resolve()
if not REPO.exists():
    REPO = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO / "results" / "framework" / "dataset_registry.json"

_CWD12 = [
    "Carpetweeds", "Crabgrass", "Eclipta", "Goosegrass", "Morningglory",
    "Nutsedge", "PalmerAmaranth", "PricklySida", "Purslane", "Ragweed",
    "Sicklepod", "SpottedSpurge",
]
_CWD12_ALNUM = {re.sub(r'[^A-Za-z0-9]', '', c).lower(): c for c in _CWD12}
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
_HOLDOUT_HINTS = ("holdout",)

_HOLDOUT_STEM_DIRS = [
    REPO / "downloads" / "cottonweeddet12" / "test" / "images",
    REPO / "downloads" / "cottonweeddet12" / "valid" / "images",
    REPO / "results" / "leave4out" / "dataset_holdout" / "test" / "images",
    REPO / "results" / "leave4out" / "dataset_holdout" / "valid" / "images",
]


def _canon(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return ""
    return _CWD12_ALNUM.get(re.sub(r'[^A-Za-z0-9]', '', raw).lower(), "")


def _is_holdout(slug: str, info: dict) -> bool:
    s = (slug + " " + str(info.get("local_path") or "")).lower()
    return any(h in s for h in _HOLDOUT_HINTS)


def _find_label_dirs(slug_dir: Path) -> list:
    out = []
    for sub in (slug_dir / "labels", slug_dir / "train" / "labels",
                slug_dir / "valid" / "labels", slug_dir / "test" / "labels"):
        if sub.is_dir():
            out.append(sub)
    return out


def holdout_stems() -> set:
    stems = set()
    for d in _HOLDOUT_STEM_DIRS:
        if d.is_dir():
            for ext in _IMG_EXTS:
                for p in d.glob("*" + ext):
                    stems.add(p.stem)
    return stems


def audit(max_labels_per_slug: int = 6000) -> dict:
    reg = safe_read_json(REGISTRY_PATH) or {}
    ds = reg.get("datasets", {}) or {}

    stems = holdout_stems()
    holdout_status = {
        "stem_dirs_present": [str(d) for d in _HOLDOUT_STEM_DIRS if d.is_dir()],
        "n_holdout_stems": len(stems),
        "protection_ok": len(stems) > 0,
    }

    per_class = defaultdict(int)
    per_class_slugs = defaultdict(set)
    slug_reports = []
    for slug, info in ds.items():
        if _is_holdout(slug, info):
            continue
        lp = info.get("local_path")
        sd = Path(lp) if lp else None
        if not sd or not sd.is_dir():
            continue
        cn = info.get("class_names") or []
        if not cn:
            continue
        cid_canon = {i: _canon(c) for i, c in enumerate(cn)}
        if not any(cid_canon.values()):
            continue
        n_scanned = 0
        hits = defaultdict(int)
        for ld in _find_label_dirs(sd):
            for lf in ld.glob("*.txt"):
                if n_scanned >= max_labels_per_slug:
                    break
                n_scanned += 1
                try:
                    present = set()
                    for line in lf.read_text().splitlines():
                        tok = line.split()
                        if tok and tok[0].isdigit():
                            present.add(int(tok[0]))
                    for cid in present:
                        c = cid_canon.get(cid)
                        if c:
                            hits[c] += 1
                except Exception:
                    pass
            if n_scanned >= max_labels_per_slug:
                break
        for c, n in hits.items():
            per_class[c] += n
            per_class_slugs[c].add(slug)
        if hits:
            slug_reports.append({"slug": slug, "scanned": n_scanned,
                                 "class_hits": dict(hits)})

    return {
        "holdout": holdout_status,
        "per_class_images": dict(sorted(per_class.items(), key=lambda kv: -kv[1])),
        "per_class_n_slugs": {k: len(v) for k, v in per_class_slugs.items()},
        "species_covered": sorted(list(per_class)),
        "species_missing": sorted([c for c in _CWD12 if c not in per_class]),
        "slug_reports": slug_reports,
        "total_weed_class_instances": sum(per_class.values()),
    }


def print_audit(a: dict) -> None:
    print("=" * 64)
    print("P1 DATASET INTEGRITY AUDIT —", REGISTRY_PATH)
    print("=" * 64)
    h = a["holdout"]
    print(f"HOLDOUT STEM PROTECTION : {'OK' if h['protection_ok'] else '*** BROKEN — NO STEMS, LEAK RISK ***'}")
    print(f"  n_holdout_stems       : {h['n_holdout_stems']}")
    print(f"  stem dirs present     : {h['stem_dirs_present'] or 'NONE'}")
    print("\nPER-CLASS trainable image counts (label files containing the class):")
    for c, n in a["per_class_images"].items():
        print(f"  {c:16s} {n:>7,}  ({a['per_class_n_slugs'].get(c,0)} slugs)")
    print(f"\nspecies covered : {len(a['species_covered'])}/12  {a['species_covered']}")
    print(f"species MISSING : {a['species_missing']}")
    print(f"total weed class-instances: {a['total_weed_class_instances']:,}")
    print("\nper-slug:")
    for r in a["slug_reports"]:
        print(f"  {r['slug'][:42]:42s} scanned={r['scanned']:>5} {r['class_hits']}")
    print("=" * 64)


def mark_overall_test() -> dict:
    from . import db as _db
    dbh = _db._get_db()
    if dbh is None:
        print(f"ERROR: Mongo unavailable ({_db.ping().get('error')})", file=sys.stderr)
        sys.exit(2)
    reg = safe_read_json(REGISTRY_PATH) or {}
    ds = reg.get("datasets", {}) or {}
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    flagged = []
    for slug, info in ds.items():
        if _is_holdout(slug, info):
            dbh[_db.COLL_SLUGS].update_one(
                {"_id": slug},
                {"$set": {"is_overall_test": True, "cwd12_eval_holdout": True,
                          "provenance": "gold", "updated_at": now}},
                upsert=True)
            flagged.append(slug)

    train_eligible = list(dbh[_db.COLL_SLUGS].find(
        {"is_overall_test": {"$ne": True}, "cwd12_eval_holdout": {"$ne": True}},
        {"_id": 1}))
    train_ids = [d["_id"] for d in train_eligible]
    leak = [s for s in flagged if s in train_ids]

    res = {"flagged_overall_test": flagged, "n_train_eligible": len(train_ids),
           "train_eligible": train_ids, "leak": leak, "leak_ok": len(leak) == 0}
    print(f"[mark] flagged overall-test: {flagged}")
    print(f"[mark] train-eligible slugs ({len(train_ids)}): {train_ids}")
    print(f"[mark] LEAK CHECK: {'CLEAN ✓' if res['leak_ok'] else 'LEAK '+str(leak)}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="P1 dataset integrity audit")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--mark-overall-test", action="store_true")
    args = ap.parse_args()
    if not (args.audit or args.mark_overall_test):
        ap.error("pass --audit and/or --mark-overall-test")
    if args.audit:
        print_audit(audit())
    if args.mark_overall_test:
        mark_overall_test()


if __name__ == "__main__":
    main()
