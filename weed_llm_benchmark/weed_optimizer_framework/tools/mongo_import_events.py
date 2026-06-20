"""
mongo_import_events.py — replay a labeling_events.jsonl into the Mongo `labeling`
collection (idempotent). Used when migrating the source-of-truth to a new server
(e.g. cluster → lab server): labeling_tracker dual-writes Mongo + JSONL, so on a
fresh Mongo we replay the JSONL once to restore the lifecycle history.

  python -m weed_optimizer_framework.tools.mongo_import_events \
      --events results/framework/labeling_events.jsonl

Dedup key = (ts, event, slug, image) so re-running is safe.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

COLL_LABELING = "labeling"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=os.path.join(
        os.environ.get("REPO_ROOT", "."),
        "results", "framework", "labeling_events.jsonl"))
    args = ap.parse_args()

    p = Path(args.events)
    if not p.is_file():
        print(f"no events file: {p}")
        return 1

    try:
        from weed_optimizer_framework.tools import db as _db
        dbh = _db._get_db()
    except Exception as e:
        print(f"mongo unavailable: {e}")
        return 2
    if dbh is None:
        print("mongo unavailable (db._get_db returned None)")
        return 2

    coll = dbh[COLL_LABELING]
    n_read = n_ins = n_dup = 0
    for line in p.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            doc = json.loads(line)
        except Exception:
            continue
        n_read += 1
        key = {"ts": doc.get("ts"), "event": doc.get("event"),
               "slug": doc.get("slug"), "image": doc.get("image")}
        try:
            if coll.find_one(key):
                n_dup += 1
                continue
            coll.insert_one(dict(doc))
            n_ins += 1
        except Exception as e:
            print(f"insert err: {e}")
    print(f"read={n_read} inserted={n_ins} skipped_dup={n_dup} "
          f"total_in_coll={coll.count_documents({})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
