"""
backfill_mongo.py — MongoDB migration Phase 4 (idempotent backfill) + registry audit.

Two jobs, one script so the audit and the load share the exact same parsing:

  --audit   READ-ONLY. Parse dataset_registry.json + class_topic_overrides.json
            and print coverage stats: per-topic image totals, weed-bbox total,
            distinct species, and the gap to the 50K real-weed-bbox north star
            (see feedback_polaris_scale). Touches NO database. Run this first
            ("先别灌数据") to understand what we have before loading it.

  --apply   Upsert the registry into Mongo `slugs` + `classes` (+ `registry_meta`
            singleton + one `audit_trail` event). Idempotent: keyed by _id, so
            re-running is safe and converges. Reuses tools.db's connection
            (Mongo must be up — see run_mongo_node.sh).

  (both flags together: audit THEN apply.)

Field fidelity: each slug doc is the registry entry copied VERBATIM, plus
`_id` (=slug) and `updated_at`. No registry field is dropped, so whatever the
live harvest added (topic, bucket, class_names_source, …) is preserved.

Usage on the cluster (where the real 52MB registry lives):
    cd $REPO
    python -m weed_optimizer_framework.tools.backfill_mongo --audit
    python -m weed_optimizer_framework.tools.backfill_mongo --apply
    # verify: curl -u … <tunnel>/api/db_status  → registry_datasets > 0
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from .registry_lock import safe_read_json

# --- paths (match dashboard_server / db.py) ---------------------------------
REPO = Path(os.environ.get(
    "REPO_ROOT",
    "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark",
)).resolve()
if not REPO.exists():
    REPO = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO / "results" / "framework" / "dataset_registry.json"
OVERRIDES_PATH = Path(os.environ.get(
    "CLASS_TOPIC_OVERRIDES_FILE",
    str(REPO / "results" / "framework" / "class_topic_overrides.json"),
))

NORTH_STAR_WEED_BBOX = 50_000  # feedback_polaris_scale / feedback_framework_invariants

# v3.0.82 (Prof directive — multi-domain extensibility): every slug/class is
# scoped to a DOMAIN = one dataset-collection agent. "weed" is the first; future
# agents (pest, crop_disease, …) just insert another `domains` doc + taxonomy,
# NO schema change. Backfilled (pre-domain) data defaults to "weed".
DEFAULT_DOMAIN = "weed"
DOMAINS_SEED = {
    "weed": {
        "_id": "weed",
        "display_name": "Weed detection",
        "description": "Field weed species detection for the laser-weeding robot.",
        "taxonomy": "cwd12",                 # canonical class taxonomy for this domain
        "target_metric": {"dataset": "cwd12_holdout",
                          "metric": "mAP50-95", "goal": 0.90},
        "harvest_queries": ["weed detection", "cotton weed", "crop and weed",
                            "grass weed", "weed dataset bounding box"],
        "status": "active",
    },
    # Future example (insert at will, no migration):
    # "pest": {"_id":"pest","display_name":"Pest detection","taxonomy":"ip102",
    #          "target_metric":{...},"harvest_queries":[...],"status":"planned"},
}

# --- canonical CWD12 (mirror of dashboard_server._CWD12 / _CWD12_ZH) --------
# After this backfill the Mongo `classes` collection becomes the single source
# of truth; this literal exists only to seed it. Keep in sync until then.
_CWD12 = [
    "Carpetweeds", "Crabgrass", "Eclipta", "Goosegrass", "Morningglory",
    "Nutsedge", "PalmerAmaranth", "PricklySida", "Purslane", "Ragweed",
    "Sicklepod", "SpottedSpurge",
]
_CWD12_ZH = {
    "Carpetweeds": "毯草", "Crabgrass": "马唐", "Eclipta": "鳢肠",
    "Goosegrass": "蟋蟀草", "Morningglory": "牵牛花", "Nutsedge": "莎草",
    "PalmerAmaranth": "苋菜 / 帕氏苋", "PricklySida": "刺苋",
    "Purslane": "马齿苋", "Ragweed": "豚草", "Sicklepod": "决明",
    "SpottedSpurge": "斑地锦",
}
_CWD12_ALNUM = {re.sub(r'[^A-Za-z0-9]', '', c).lower(): c for c in _CWD12}

# bbox-ish signals on a slug's annotation/format fields
_BBOX_ANNOT = ("bbox", "detection", "segmentation")
_BBOX_FORMAT = ("yolo", "coco", "voc_xml", "voc", "pascal")
_WEED_TOPICS = ("cwd12", "weed")
# slug that must NEVER enter training (eval contamination) — excluded from the
# trainable weed-bbox tally but still loaded into Mongo (flagged).
_HOLDOUT_HINTS = ("holdout", "cottonweed_holdout")


def _canon(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return ""
    alnum = re.sub(r'[^A-Za-z0-9]', '', raw).lower()
    if not alnum:
        return ""
    if alnum in _CWD12_ALNUM:
        return _CWD12_ALNUM[alnum]
    parts = re.split(r'[^A-Za-z0-9]+', raw)
    return "".join(p[:1].upper() + p[1:].lower() for p in parts if p)


def _is_bbox(info: dict) -> bool:
    annot = str(info.get("annotation") or "").lower()
    fmt = str(info.get("format") or "").lower()
    return any(k in annot for k in _BBOX_ANNOT) or any(k in fmt for k in _BBOX_FORMAT)


def _is_holdout(slug: str, info: dict) -> bool:
    s = (slug + " " + str(info.get("local_path") or "")).lower()
    return any(h in s for h in _HOLDOUT_HINTS)


def _imgs(info: dict) -> int:
    for k in ("local_images", "images"):
        v = info.get(k)
        if isinstance(v, int) and v > 0:
            return v
    return 0


# --------------------------------------------------------------------------- #
# AUDIT
# --------------------------------------------------------------------------- #

def audit() -> dict:
    reg = safe_read_json(REGISTRY_PATH) or {}
    ds = reg.get("datasets", {}) or {}
    overrides = safe_read_json(OVERRIDES_PATH) or {}

    topic_imgs: dict = {}
    topic_slugs: dict = {}
    weed_bbox_imgs = 0
    weed_bbox_slugs = []
    holdout_imgs = 0
    species: set = set()

    for slug, info in ds.items():
        if str(info.get("status") or "") not in ("downloaded", "known", ""):
            pass  # count all; status just informs
        n = _imgs(info)
        topic = str(info.get("topic") or "unknown").lower()
        # effective topic: registry field, else override of any class, else unknown
        if topic in ("unknown", "", "none"):
            for cn in (info.get("class_names") or []):
                ov = overrides.get(_canon(cn)) or overrides.get(cn)
                if ov:
                    topic = ov
                    break
        topic_imgs[topic] = topic_imgs.get(topic, 0) + n
        topic_slugs[topic] = topic_slugs.get(topic, 0) + 1

        bbox = _is_bbox(info)
        weedish = topic in _WEED_TOPICS
        hold = _is_holdout(slug, info)
        if bbox and weedish and hold:
            holdout_imgs += n
        if bbox and weedish and not hold and n > 0:
            weed_bbox_imgs += n
            cns = [c for c in (_canon(c) for c in (info.get("class_names") or [])) if c]
            species.update(cns)
            weed_bbox_slugs.append({
                "slug": slug, "topic": topic, "images": n,
                "annotation": info.get("annotation"), "format": info.get("format"),
                "n_classes": len(cns), "classes": cns[:8],
                "status": info.get("status"),
            })

    weed_bbox_slugs.sort(key=lambda r: -r["images"])
    return {
        "total_slugs": len(ds),
        "topic_imgs": dict(sorted(topic_imgs.items(), key=lambda kv: -kv[1])),
        "topic_slugs": topic_slugs,
        "weed_bbox_imgs": weed_bbox_imgs,
        "weed_bbox_slugs": weed_bbox_slugs,
        "holdout_imgs_excluded": holdout_imgs,
        "distinct_species": sorted(species),
        "n_distinct_species": len(species),
        "north_star": NORTH_STAR_WEED_BBOX,
        "gap_to_north_star": max(0, NORTH_STAR_WEED_BBOX - weed_bbox_imgs),
        "n_overrides": len(overrides),
    }


def print_audit(a: dict) -> None:
    print("=" * 64)
    print("REGISTRY AUDIT (read-only) —", REGISTRY_PATH)
    print("=" * 64)
    print(f"total slugs                 : {a['total_slugs']}")
    print(f"class-topic overrides       : {a['n_overrides']}")
    print("\nimages by topic (all slugs):")
    for t, n in a["topic_imgs"].items():
        print(f"  {t:12s} {n:>12,}  ({a['topic_slugs'].get(t,0)} slugs)")
    print("\n--- TRAINABLE WEED-BBOX (topic∈{cwd12,weed} ∧ bbox ∧ ¬holdout) ---")
    print(f"weed-bbox images            : {a['weed_bbox_imgs']:,}")
    print(f"holdout images (excluded)   : {a['holdout_imgs_excluded']:,}")
    print(f"distinct species (canon)    : {a['n_distinct_species']}")
    print(f"NORTH STAR                  : {a['north_star']:,}")
    print(f"GAP TO 50K                  : {a['gap_to_north_star']:,}")
    print("\nweed-bbox slugs (desc by images):")
    for r in a["weed_bbox_slugs"]:
        print(f"  {r['images']:>8,}  {r['slug'][:42]:42s} "
              f"{str(r['annotation'])[:10]:10s} cls={r['n_classes']} {r['classes']}")
    print("\ndistinct species:", ", ".join(a["distinct_species"]) or "(none)")
    print("=" * 64)


# --------------------------------------------------------------------------- #
# APPLY (backfill into Mongo)
# --------------------------------------------------------------------------- #

def apply() -> dict:
    from . import db as _db
    dbh = _db._get_db()
    if dbh is None:
        print(f"ERROR: Mongo not available ({_db.ping().get('error')}). "
              f"Start it first (run_mongo_node.sh) then retry --apply.",
              file=sys.stderr)
        sys.exit(2)

    reg = safe_read_json(REGISTRY_PATH) or {}
    ds = reg.get("datasets", {}) or {}
    overrides = safe_read_json(OVERRIDES_PATH) or {}

    # --- slugs (verbatim + _id + updated_at) ---
    n_slugs = 0
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    for slug, info in ds.items():
        doc = dict(info)
        doc.pop("_id", None)
        doc["updated_at"] = now
        doc.setdefault("domain", DEFAULT_DOMAIN)   # v3.0.82 multi-domain scope
        if _is_holdout(slug, info):
            doc["cwd12_eval_holdout"] = True  # enforce NEVER-TRAIN at query time
        dbh[_db.COLL_SLUGS].update_one({"_id": slug}, {"$set": doc}, upsert=True)
        n_slugs += 1

    # --- domains: one doc per dataset-collection agent (multi-domain seed) ---
    n_domains = 0
    for dom_id, dom_doc in DOMAINS_SEED.items():
        dbh[_db.COLL_DOMAINS].update_one({"_id": dom_id}, {"$set": dom_doc},
                                         upsert=True)
        n_domains += 1

    # --- classes: CWD12 canon + every species seen in slugs + overrides ---
    # v3.0.82: generalize taxonomy. `taxonomies` is the extensible form (a class
    # can belong to several taxonomies across domains); cwd12_index/is_cwd12 are
    # kept for back-compat with existing readers.
    classes: dict = {}
    for i, c in enumerate(_CWD12):
        classes[c] = {"_id": c, "topic": "cwd12", "is_cwd12": True,
                      "cwd12_index": i, "cn_zh": _CWD12_ZH.get(c, ""),
                      "domain": "weed",
                      "taxonomies": [{"taxonomy": "cwd12", "index": i}]}
    for info in ds.values():
        for cn in (info.get("class_names") or []):
            canon = _canon(cn)
            if not canon or canon in classes:
                continue
            classes[canon] = {"_id": canon, "is_cwd12": False, "domain": "weed"}
    # apply topic overrides on top
    for cls, topic in overrides.items():
        canon = _canon(cls) or cls
        classes.setdefault(canon, {"_id": canon, "is_cwd12": canon in _CWD12})
        classes[canon]["topic"] = topic
    n_classes = 0
    for canon, doc in classes.items():
        dbh[_db.COLL_CLASSES].update_one({"_id": canon}, {"$set": doc}, upsert=True)
        n_classes += 1

    # --- registry_meta singleton (so db.get_registry reconstructs disc/total) ---
    dbh["registry_meta"].update_one(
        {"_id": "singleton"},
        {"$set": {"discovered": reg.get("discovered", []),
                  "total_downloaded": reg.get("total_downloaded", 0),
                  "updated_at": now}},
        upsert=True)

    # --- audit_trail event ---
    dbh[_db.COLL_AUDIT].insert_one({
        "ts": now, "actor": "system", "event": "mongo.backfill",
        "target": {"kind": "registry", "id": str(REGISTRY_PATH)},
        "after": {"slugs": n_slugs, "classes": n_classes, "domains": n_domains},
        "reason": "phase4_backfill",
    })

    res = {"slugs_upserted": n_slugs, "classes_upserted": n_classes,
           "domains_upserted": n_domains,
           "slugs_in_mongo": dbh[_db.COLL_SLUGS].estimated_document_count(),
           "classes_in_mongo": dbh[_db.COLL_CLASSES].estimated_document_count(),
           "domains_in_mongo": dbh[_db.COLL_DOMAINS].estimated_document_count()}
    print(f"[apply] upserted {n_slugs} slugs, {n_classes} classes, "
          f"{n_domains} domains")
    print(f"[apply] mongo now: slugs={res['slugs_in_mongo']} "
          f"classes={res['classes_in_mongo']} domains={res['domains_in_mongo']}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Mongo backfill + registry audit")
    ap.add_argument("--audit", action="store_true", help="read-only stats")
    ap.add_argument("--apply", action="store_true", help="upsert into Mongo")
    args = ap.parse_args()
    if not (args.audit or args.apply):
        ap.error("pass --audit and/or --apply")
    if args.audit:
        print_audit(audit())
    if args.apply:
        apply()


if __name__ == "__main__":
    main()
