"""
db.py — Mongo-first / JSON-fallback data access layer (MongoDB migration Phase 1-2).

This is the single seam between the app and "where the dataset metadata lives."
Per Prof Zhang (2026-05-28) the labeler must use MongoDB; per
`docs/mongodb_schema.md` the migration is incremental and reversible. This
module implements migration **step 2 — the read path**:

    "introduce a tools/db.py wrapper that reads from Mongo *first* and falls
     back to the JSON registry. New endpoints use Mongo only."

Design goals
------------
1. **Never break the running app.** If pymongo is missing, the connection
   string is unset, or Mongo is down, every read transparently falls back to
   the existing JSON files (`dataset_registry.json`,
   `class_topic_overrides.json`). So this module is safe to import and call
   on the Mac (no Mongo) and on the cluster *before* Mongo is stood up.
2. **Shape-compatible.** `get_registry()` returns the SAME dict shape as
   `dataset_registry.json` (`{"datasets": {...}, "discovered": [...],
   "total_downloaded": N}`) so existing callers can swap with zero downstream
   changes.
3. **Cheap to probe.** Connection uses a short serverSelectionTimeout and the
   result of `available()` is cached, so a dead Mongo doesn't add latency to
   every request.

Connection string resolution (first hit wins)
---------------------------------------------
1. env `AGENTAI_MONGO_URL`
2. secret file `~/.mongo_url` (one line, e.g. `mongodb://127.0.0.1:27017/agentai`)
   — same secret-file pattern as the Roboflow key / GH PAT.
3. none → fallback mode (JSON only). `available()` returns False.

DB name defaults to the path component of the URL, else `AGENTAI_MONGO_DB`,
else `agentai`.

Public read API
---------------
    db.available() -> bool
    db.get_registry() -> dict            # dataset_registry.json shape
    db.get_slug(slug) -> dict | None
    db.list_slugs(topic=None, status=None, bucket=None) -> list[dict]
    db.get_class_topic_overrides() -> dict   # {class_name: topic}
    db.list_classes() -> list[dict]
    db.ping() -> dict                    # diagnostics for /api/db_status

Write path (slug upsert, audit_trail, dual-write) is Phase 3 — intentionally
NOT here yet. This file is read-only on purpose so it can land without risk.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

# Reuse the project's crash-safe JSON reader for the fallback path so we match
# existing behavior exactly (retry on torn writes from concurrent writers).
try:
    from .registry_lock import safe_read_json
except Exception:  # pragma: no cover - allows standalone import in odd setups
    import json as _json

    def safe_read_json(path, retries=5, retry_sleep=0.2):
        try:
            with open(path) as f:
                return _json.load(f)
        except FileNotFoundError:
            return None
        except Exception:
            return None


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

# Match dashboard_server.REPO resolution so fallback files line up on cluster
# and locally.
REPO = Path(os.environ.get(
    "REPO_ROOT",
    "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark",
)).resolve()

# Local-checkout fallback: if the cluster path doesn't exist (running on the
# Mac), use the repo this file lives in.
if not REPO.exists():
    REPO = Path(__file__).resolve().parents[2]

REGISTRY_PATH = REPO / "results" / "framework" / "dataset_registry.json"
CLASS_TOPIC_OVERRIDES_FILE = Path(os.environ.get(
    "CLASS_TOPIC_OVERRIDES_FILE",
    str(REPO / "results" / "framework" / "class_topic_overrides.json"),
))

DEFAULT_DB_NAME = os.environ.get("AGENTAI_MONGO_DB", "agentai")
_SECRET_FILE = Path(os.path.expanduser("~/.mongo_url"))

# Collection names (mirror docs/mongodb_schema.md).
COLL_SLUGS = "slugs"
COLL_CLASSES = "classes"
COLL_IMAGES = "images"
COLL_EXEMPLARS = "exemplars"
COLL_AGENT_TASKS = "agent_tasks"
COLL_AUDIT = "audit_trail"

# Short timeout so a dead Mongo never stalls a request for more than this.
_CONNECT_TIMEOUT_MS = int(os.environ.get("AGENTAI_MONGO_TIMEOUT_MS", "800"))


# --------------------------------------------------------------------------- #
# Connection (lazy singleton, thread-safe, probe cached)
# --------------------------------------------------------------------------- #

_lock = threading.Lock()
_state: dict = {
    "tried": False,      # have we attempted a connection this process?
    "client": None,      # MongoClient or None
    "db": None,          # Database handle or None
    "url": None,         # resolved connection string
    "error": None,       # last connection error string (for diagnostics)
}


def _resolve_url() -> Optional[str]:
    url = os.environ.get("AGENTAI_MONGO_URL")
    if url:
        return url.strip()
    try:
        if _SECRET_FILE.is_file():
            txt = _SECRET_FILE.read_text().strip()
            if txt:
                return txt
    except Exception:
        pass
    return None


def _db_name_from_url(url: str) -> str:
    # mongodb://host:port/<dbname>?opts  → dbname (if present)
    try:
        tail = url.split("://", 1)[1]
        after_host = tail.split("/", 1)
        if len(after_host) == 2 and after_host[1]:
            name = after_host[1].split("?", 1)[0].strip("/")
            if name:
                return name
    except Exception:
        pass
    return DEFAULT_DB_NAME


def _connect() -> None:
    """Attempt a connection once. Idempotent; subsequent calls are no-ops
    unless reset() is called. Sets _state['db'] to a live handle or None."""
    if _state["tried"]:
        return
    _state["tried"] = True
    url = _resolve_url()
    _state["url"] = url
    if not url:
        _state["error"] = "no connection string (AGENTAI_MONGO_URL / ~/.mongo_url unset)"
        return
    try:
        from pymongo import MongoClient
    except Exception as e:
        _state["error"] = f"pymongo not installed: {e}"
        return
    try:
        client = MongoClient(url, serverSelectionTimeoutMS=_CONNECT_TIMEOUT_MS)
        # Force a round-trip so we know it's actually reachable now.
        client.admin.command("ping")
        _state["client"] = client
        _state["db"] = client[_db_name_from_url(url)]
        _state["error"] = None
    except Exception as e:
        _state["client"] = None
        _state["db"] = None
        _state["error"] = f"connect/ping failed: {e}"


def reset() -> None:
    """Drop cached connection state (e.g. after standing Mongo up, to retry)."""
    with _lock:
        cl = _state.get("client")
        if cl is not None:
            try:
                cl.close()
            except Exception:
                pass
        _state.update({"tried": False, "client": None, "db": None,
                       "url": None, "error": None})


def _get_db():
    """Return a live Database handle, or None if Mongo is unavailable."""
    if _state["db"] is not None:
        return _state["db"]
    with _lock:
        _connect()
    return _state["db"]


def available() -> bool:
    """True iff a live Mongo connection is usable right now."""
    return _get_db() is not None


def ping() -> dict:
    """Diagnostics for a /api/db_status endpoint. Never raises."""
    db = _get_db()
    info = {
        "available": db is not None,
        "url": _redact(_state.get("url")),
        "db_name": db.name if db is not None else None,
        "error": _state.get("error"),
        "fallback": "json" if db is None else None,
    }
    if db is not None:
        try:
            info["counts"] = {
                COLL_SLUGS: db[COLL_SLUGS].estimated_document_count(),
                COLL_CLASSES: db[COLL_CLASSES].estimated_document_count(),
            }
        except Exception as e:
            info["counts_error"] = str(e)
    return info


def _redact(url: Optional[str]) -> Optional[str]:
    """Hide credentials in a mongodb URL for safe display."""
    if not url:
        return url
    try:
        if "@" in url and "://" in url:
            scheme, rest = url.split("://", 1)
            creds, host = rest.split("@", 1)
            return f"{scheme}://***@{host}"
    except Exception:
        pass
    return url


# --------------------------------------------------------------------------- #
# Read API — Mongo first, JSON fallback
# --------------------------------------------------------------------------- #

def _registry_from_json() -> dict:
    data = safe_read_json(REGISTRY_PATH)
    if not isinstance(data, dict):
        return {"datasets": {}, "discovered": [], "total_downloaded": 0}
    data.setdefault("datasets", {})
    data.setdefault("discovered", [])
    data.setdefault("total_downloaded", 0)
    return data


def get_registry() -> dict:
    """Return the full registry in dataset_registry.json shape.

    Mongo path: reconstruct `datasets` from the `slugs` collection (each doc's
    `_id` is the slug). JSON path: read the file. On any Mongo error, fall
    back to JSON so the caller always gets a usable dict.
    """
    db = _get_db()
    if db is None:
        return _registry_from_json()
    try:
        datasets: dict = {}
        total_downloaded = 0
        for doc in db[COLL_SLUGS].find({}):
            slug = doc.pop("_id")
            # Drop Mongo-only bookkeeping that isn't part of the JSON contract.
            doc.pop("updated_at", None)
            datasets[slug] = doc
            if doc.get("status") == "downloaded":
                total_downloaded += 1
        meta = db["registry_meta"].find_one({"_id": "singleton"}) or {}
        return {
            "datasets": datasets,
            "discovered": meta.get("discovered", []),
            "total_downloaded": meta.get("total_downloaded", total_downloaded),
        }
    except Exception as e:
        _state["error"] = f"get_registry mongo read failed, used json: {e}"
        return _registry_from_json()


def get_slug(slug: str) -> Optional[dict]:
    """Return one slug's metadata doc, or None."""
    db = _get_db()
    if db is not None:
        try:
            doc = db[COLL_SLUGS].find_one({"_id": slug})
            if doc is not None:
                return doc
            # fall through to JSON if not in Mongo yet (mid-migration)
        except Exception:
            pass
    return _registry_from_json()["datasets"].get(slug)


def list_slugs(topic: Optional[str] = None,
               status: Optional[str] = None,
               bucket: Optional[str] = None) -> list:
    """Return slug docs (with `slug` key) optionally filtered. JSON fallback
    filters the same fields in Python."""
    db = _get_db()
    if db is not None:
        try:
            q: dict = {}
            if topic:
                q["topic"] = topic
            if status:
                q["status"] = status
            if bucket:
                q["bucket"] = bucket
            out = []
            for doc in db[COLL_SLUGS].find(q):
                doc["slug"] = doc.pop("_id")
                out.append(doc)
            return out
        except Exception:
            pass
    out = []
    for slug, info in _registry_from_json()["datasets"].items():
        if topic and info.get("topic") != topic:
            continue
        if status and info.get("status") != status:
            continue
        if bucket and info.get("bucket") != bucket:
            continue
        d = dict(info)
        d["slug"] = slug
        out.append(d)
    return out


def get_class_topic_overrides() -> dict:
    """Return {class_name: topic}. Mongo path reads the `classes` collection
    (docs carrying an explicit `topic`); JSON path reads
    class_topic_overrides.json via class_topic_store semantics."""
    db = _get_db()
    if db is not None:
        try:
            out = {}
            for doc in db[COLL_CLASSES].find({"topic": {"$exists": True}},
                                             {"_id": 1, "topic": 1}):
                out[doc["_id"]] = doc["topic"]
            if out:
                return out
            # empty collection mid-migration → fall back to JSON
        except Exception:
            pass
    data = safe_read_json(CLASS_TOPIC_OVERRIDES_FILE)
    return data if isinstance(data, dict) else {}


def list_classes() -> list:
    """Return canonical class docs from the `classes` collection. JSON has no
    equivalent canonical store yet, so the fallback derives a minimal list
    from the topic-overrides keys (best effort)."""
    db = _get_db()
    if db is not None:
        try:
            return list(db[COLL_CLASSES].find({}))
        except Exception:
            pass
    return [{"_id": cls, "topic": topic}
            for cls, topic in get_class_topic_overrides().items()]


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

def _self_test() -> None:
    """Run as:  python -m weed_optimizer_framework.tools.db
    Prints connection status + a registry summary using whichever backend is
    live (Mongo if up, else JSON)."""
    print(f"REPO            : {REPO}")
    print(f"registry path   : {REGISTRY_PATH}  exists={REGISTRY_PATH.exists()}")
    print(f"mongo available : {available()}")
    p = ping()
    print(f"ping            : {p}")
    reg = get_registry()
    ds = reg.get("datasets", {})
    print(f"backend in use  : {'mongo' if available() else 'json-fallback'}")
    print(f"datasets        : {len(ds)} slugs, "
          f"total_downloaded={reg.get('total_downloaded')}")
    with_cn = sum(1 for v in ds.values() if v.get("class_names"))
    print(f"slugs w/ class_names: {with_cn}")
    if ds:
        first = next(iter(ds))
        print(f"sample slug     : {first} -> classes={ds[first].get('classes')}, "
              f"class_names[:3]={ (ds[first].get('class_names') or [])[:3] }")
    ov = get_class_topic_overrides()
    print(f"topic overrides : {len(ov)} entries")


if __name__ == "__main__":
    _self_test()
