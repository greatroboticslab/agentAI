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
COLL_DOMAINS = "domains"          # v3.0.82: one doc per dataset-collection agent
COLL_USERS = "users"              # v3.0.129: one doc per user (Prof: per-student login + attribution)

# v3.0.82 (Prof directive — multi-domain extensibility): every slug/class is
# scoped to a DOMAIN (= a dataset-collection agent's target). "weed" is just
# the first; future agents (pest, crop_disease, …) insert a `domains` doc +
# their taxonomy with NO schema change. Default for un-tagged/backfilled data.
DEFAULT_DOMAIN = os.environ.get("AGENTAI_DEFAULT_DOMAIN", "weed")

# Short timeout so a dead Mongo never stalls a request for more than this.
# v3.0.80: 1500ms (was 800) — Mongo may now be cross-node (harvest/trainer jobs
# reach the dashboard node's mongod over the cluster net), slightly slower than
# localhost. Still bounded so a dead Mongo never hangs a request.
_CONNECT_TIMEOUT_MS = int(os.environ.get("AGENTAI_MONGO_TIMEOUT_MS", "1500"))


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


def get_registry(domain: Optional[str] = None) -> dict:
    """Return the full registry in dataset_registry.json shape.

    Mongo path: reconstruct `datasets` from the `slugs` collection (each doc's
    `_id` is the slug). JSON path: read the file. On any Mongo error, fall
    back to JSON so the caller always gets a usable dict.

    `domain` (v3.0.82): if given, return only slugs for that collection-agent
    domain (e.g. "weed"). JSON fallback filters in Python (treats a missing
    `domain` field as DEFAULT_DOMAIN for back-compat).
    """
    db = _get_db()
    if db is None:
        reg = _registry_from_json()
        if domain:
            reg["datasets"] = {
                s: i for s, i in reg["datasets"].items()
                if (i.get("domain") or DEFAULT_DOMAIN) == domain}
        return reg
    try:
        datasets: dict = {}
        total_downloaded = 0
        q = {"domain": domain} if domain else {}
        for doc in db[COLL_SLUGS].find(q):
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
               bucket: Optional[str] = None,
               domain: Optional[str] = None) -> list:
    """Return slug docs (with `slug` key) optionally filtered. JSON fallback
    filters the same fields in Python. `domain` scopes to one collection agent."""
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
            if domain:
                q["domain"] = domain
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
        if domain and (info.get("domain") or DEFAULT_DOMAIN) != domain:
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


def list_classes(domain: Optional[str] = None) -> list:
    """Return canonical class docs from the `classes` collection. JSON has no
    equivalent canonical store yet, so the fallback derives a minimal list
    from the topic-overrides keys (best effort). `domain` scopes to one agent."""
    db = _get_db()
    if db is not None:
        try:
            q = {"domain": domain} if domain else {}
            return list(db[COLL_CLASSES].find(q))
        except Exception:
            pass
    return [{"_id": cls, "topic": topic}
            for cls, topic in get_class_topic_overrides().items()]


def get_domains() -> list:
    """Return all dataset-collection-agent domain docs (v3.0.82). Each doc:
    {_id, display_name, taxonomy, target_metric, harvest_queries, status, ...}.
    Returns [] when Mongo is down (JSON has no domains store)."""
    db = _get_db()
    if db is None:
        return []
    try:
        return list(db[COLL_DOMAINS].find({}))
    except Exception:
        return []


def get_domain(domain_id: str) -> Optional[dict]:
    """Return one domain doc, or None."""
    db = _get_db()
    if db is None:
        return None
    try:
        return db[COLL_DOMAINS].find_one({"_id": domain_id})
    except Exception:
        return None


# v3.0.137: generalized agent schema. An agent (domain) is no longer implicitly
# image+YOLO+detection — it declares its MODALITY (what kind of data), TASK (what
# it learns), and MODEL. Defaults keep every pre-v3.0.137 domain (incl. weed)
# behaving exactly as before (image / detection / yolo).
TASKS = ["detection", "classification", "segmentation", "pose", "tracking",
         "rl_policy", "ssl_pretrain"]
MODALITIES = ["image", "video", "sensor", "pointcloud", "audio", "text"]
# sensible default headline metric per task
_TASK_DEFAULT_METRIC = {
    "detection": "mAP50-95", "classification": "top1_accuracy",
    "segmentation": "mIoU", "pose": "PCK", "tracking": "MOTA",
    "rl_policy": "success_rate", "ssl_pretrain": "linear_probe_acc",
}


def delete_domain(domain_id: str, actor: str = "user") -> bool:
    """v3.0.144: remove an agent (domain) doc. Returns True if deleted."""
    db = _get_db()
    if db is None:
        return False
    try:
        r = db[COLL_DOMAINS].delete_one({"_id": domain_id})
        if r.deleted_count:
            try:
                db[COLL_AUDIT].insert_one({"ts": _now(), "actor": actor,
                    "event": "domain.delete", "target": {"kind": "domain", "id": domain_id}})
            except Exception:
                pass
        return bool(r.deleted_count)
    except Exception:
        return False


def update_domain(domain_id: str, fields: dict, actor: str = "user") -> Optional[dict]:
    """v3.0.144: update an agent's editable config (display_name, task, modality,
    model, harvest_queries, n_subagents). Returns the updated doc or None."""
    db = _get_db()
    if db is None:
        return None
    allowed = {"display_name", "task", "modality", "model", "harvest_queries",
               "n_subagents", "target_metric", "status"}
    sets = {k: v for k, v in (fields or {}).items() if k in allowed}
    if not sets:
        return get_domain(domain_id)
    try:
        r = db[COLL_DOMAINS].update_one({"_id": domain_id}, {"$set": sets})
        if r.matched_count == 0:
            return None
        try:
            db[COLL_AUDIT].insert_one({"ts": _now(), "actor": actor,
                "event": "domain.update", "target": {"kind": "domain", "id": domain_id},
                "after": sets})
        except Exception:
            pass
        return get_domain(domain_id)
    except Exception:
        return None


def create_domain(domain_id: str, display_name: str, taxonomy=None,
                  target_metric: str = None, harvest_queries=None,
                  n_subagents: int = 2, status: str = "created",
                  task: str = "detection", modality=None,
                  model: str = "auto", owner: str = "") -> Optional[dict]:
    """v3.0.107/137: insert a new agent (domain) doc. Additive config — a domain
    declares task / modality / model so the platform generalizes beyond weed.
    Returns the created doc, "exists" if taken, or None if Mongo is down."""
    db = _get_db()
    if db is None:
        return None
    try:
        if db[COLL_DOMAINS].find_one({"_id": domain_id}):
            return "exists"
        task = task if task in TASKS else "detection"
        mods = [m for m in (modality or ["image"]) if m in MODALITIES] or ["image"]
        if not target_metric:
            target_metric = _TASK_DEFAULT_METRIC.get(task, "mAP50-95")
        doc = {
            "_id": domain_id,
            "display_name": display_name,
            "taxonomy": list(taxonomy or []),
            "target_metric": target_metric,
            "harvest_queries": list(harvest_queries or []),
            "n_subagents": int(n_subagents),
            "status": status,
            "task": task,
            "modality": mods,
            "model": model or "auto",
            "owner": owner or "",
            "created_at": _now(),
        }
        db[COLL_DOMAINS].insert_one(doc)
        return doc
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Write API — Phase 3 dual-write (Mongo AND JSON, both authoritative)
# --------------------------------------------------------------------------- #
#
# Migration step 3 ("dual-write"): when a slug is added/updated, write to BOTH
# Mongo and the JSON registry so neither is stale during the cutover. Mongo is
# best-effort: if it's down, the JSON write still happens and the next backfill
# reconciles. JSON write uses registry_lock.atomic_write_json (crash-safe on
# Lustre). These are what the harvester calls so "new harvest goes straight
# into Mongo" without abandoning the JSON path yet.

def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)


def upsert_slug(slug: str, fields: dict, actor: str = "system") -> dict:
    """Add/update one slug in BOTH Mongo `slugs` and the JSON registry.

    `fields` is merged onto any existing entry (partial update). Returns
    {"mongo": bool, "json": bool} indicating which backends were written.
    Records a Mongo audit_trail event when Mongo is available.
    """
    if not slug or "/" in slug or slug in (".", ".."):
        raise ValueError(f"unsafe slug: {slug!r}")
    fields = dict(fields or {})
    fields.pop("_id", None)
    wrote = {"mongo": False, "json": False}

    # --- Mongo (best effort) ---
    dbh = _get_db()
    if dbh is not None:
        try:
            before = dbh[COLL_SLUGS].find_one({"_id": slug})
            doc = dict(fields)
            doc["updated_at"] = _now()
            dbh[COLL_SLUGS].update_one({"_id": slug}, {"$set": doc}, upsert=True)
            wrote["mongo"] = True
            try:
                dbh[COLL_AUDIT].insert_one({
                    "ts": _now(), "actor": actor, "event": "slug.upsert",
                    "target": {"kind": "slug", "id": slug},
                    "before": {k: before.get(k) for k in fields} if before else None,
                    "after": fields, "reason": "dual_write",
                })
            except Exception:
                pass
        except Exception as e:
            _state["error"] = f"upsert_slug mongo write failed: {e}"

    # --- JSON registry (authoritative until cutover) ---
    try:
        from .registry_lock import atomic_write_json
        reg = _registry_from_json()
        cur = reg["datasets"].get(slug, {})
        cur.update(fields)
        reg["datasets"][slug] = cur
        reg["total_downloaded"] = sum(
            1 for v in reg["datasets"].values() if v.get("status") == "downloaded")
        atomic_write_json(REGISTRY_PATH, reg)
        wrote["json"] = True
    except Exception as e:
        _state["error"] = f"upsert_slug json write failed: {e}"

    return wrote


def set_class_topic(cls: str, topic: str, actor: str = "user") -> dict:
    """Set a class→topic in BOTH Mongo `classes` and class_topic_overrides.json.
    `topic='_clear_'` removes the override. Returns {"mongo":bool,"json":bool}."""
    wrote = {"mongo": False, "json": False}

    dbh = _get_db()
    if dbh is not None:
        try:
            if topic == "_clear_":
                dbh[COLL_CLASSES].update_one({"_id": cls}, {"$unset": {"topic": ""}})
            else:
                dbh[COLL_CLASSES].update_one(
                    {"_id": cls}, {"$set": {"topic": topic}}, upsert=True)
            wrote["mongo"] = True
            try:
                dbh[COLL_AUDIT].insert_one({
                    "ts": _now(), "actor": actor, "event": "class.topic_set",
                    "target": {"kind": "class", "id": cls},
                    "after": {"topic": topic}, "reason": "dual_write",
                })
            except Exception:
                pass
        except Exception as e:
            _state["error"] = f"set_class_topic mongo write failed: {e}"

    try:
        from .class_topic_store import save_override
        wrote["json"] = bool(save_override(cls, topic))
    except Exception as e:
        _state["error"] = f"set_class_topic json write failed: {e}"

    return wrote


_BSON_INT64_MAX = 2 ** 63 - 1


def _bson_safe(v):
    """Recursively make a value BSON-safe. MongoDB ints must fit in int64; some
    registry fields (e.g. unsigned dHash values) exceed it → 'can only handle up
    to 8-byte ints'. Convert oversized ints to str so the mirror never crashes."""
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return str(v) if abs(v) > _BSON_INT64_MAX else v
    if isinstance(v, dict):
        return {k: _bson_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_bson_safe(x) for x in v]
    return v


def mirror_registry_to_mongo(registry: dict) -> dict:
    """Mongo-ONLY mirror of a full registry dict (the harvester already wrote
    JSON; this just keeps Mongo in sync at the same chokepoint). Best-effort:
    never raises, returns {"ok":bool, "slugs":n}. Upserts every slug + the
    registry_meta singleton. Safe to call on every _save_registry."""
    dbh = _get_db()
    if dbh is None:
        return {"ok": False, "slugs": 0, "reason": "mongo unavailable"}
    try:
        ds = (registry or {}).get("datasets", {}) or {}
        n = 0
        now = _now()
        for slug, info in ds.items():
            doc = dict(info)
            doc.pop("_id", None)
            # dhash_cache = large per-image hash cache (not source-of-truth, and
            # its unsigned-64bit values overflow BSON int64) → never mirror it.
            doc.pop("dhash_cache", None)
            doc = _bson_safe(doc)
            doc["updated_at"] = now
            doc.setdefault("domain", DEFAULT_DOMAIN)  # v3.0.82 multi-domain
            dbh[COLL_SLUGS].update_one({"_id": slug}, {"$set": doc}, upsert=True)
            n += 1
        dbh["registry_meta"].update_one(
            {"_id": "singleton"},
            {"$set": {"discovered": registry.get("discovered", []),
                      "total_downloaded": registry.get("total_downloaded", 0),
                      "updated_at": now}}, upsert=True)
        return {"ok": True, "slugs": n}
    except Exception as e:
        _state["error"] = f"mirror_registry_to_mongo failed: {e}"
        return {"ok": False, "slugs": 0, "reason": str(e)}


def log_audit(event: str, target: dict, after: dict = None,
              before: dict = None, reason: str = "", actor: str = "system") -> bool:
    """Append one event to the Mongo audit_trail. No-op (returns False) if Mongo
    is down — audit_trail is Mongo-only (no JSON equivalent)."""
    dbh = _get_db()
    if dbh is None:
        return False
    try:
        dbh[COLL_AUDIT].insert_one({
            "ts": _now(), "actor": actor, "event": event,
            "target": target, "before": before, "after": after, "reason": reason,
        })
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Users (v3.0.129) — Prof Zhang: students log in with their own account; we save
# users + track who uploaded what. Mongo-backed (best-effort, like domains).
# A user doc: {_id: user_id, email, name, role, auth_provider, created_at, last_seen}.
# --------------------------------------------------------------------------- #

def list_users() -> list:
    db = _get_db()
    if db is None:
        return []
    try:
        return list(db[COLL_USERS].find({}).sort("last_seen", -1))
    except Exception:
        return []


def get_user(user_id: str) -> Optional[dict]:
    db = _get_db()
    if db is None:
        return None
    try:
        return db[COLL_USERS].find_one({"_id": user_id})
    except Exception:
        return None


def upsert_user(user_id: str, email: str = "", name: str = "",
                role: str = "member", auth_provider: str = "basic") -> Optional[dict]:
    """Create the user if new (stamping created_at + role/email/name), and always
    bump last_seen. Idempotent — safe to call on every authenticated action.
    Returns the user doc, or None if Mongo is down."""
    if not user_id:
        return None
    db = _get_db()
    if db is None:
        return None
    try:
        now = _now()
        existing = db[COLL_USERS].find_one({"_id": user_id})
        if existing is None:
            doc = {
                "_id": user_id, "email": email, "name": name or user_id,
                "role": role, "auth_provider": auth_provider,
                "created_at": now, "last_seen": now,
            }
            db[COLL_USERS].insert_one(doc)
            return doc
        # update last_seen + fill any newly-provided identity fields
        sets = {"last_seen": now}
        if email and not existing.get("email"):
            sets["email"] = email
        if name and (not existing.get("name") or existing.get("name") == user_id):
            sets["name"] = name
        if auth_provider and existing.get("auth_provider") in (None, "basic"):
            sets["auth_provider"] = auth_provider
        db[COLL_USERS].update_one({"_id": user_id}, {"$set": sets})
        existing.update(sets)
        return existing
    except Exception:
        return None


def create_user(user_id: str, email: str = "", name: str = "",
                role: str = "member", auth_provider: str = "basic") -> Optional[dict]:
    """Insert a user; returns the doc, "exists" if taken, or None if Mongo down."""
    db = _get_db()
    if db is None:
        return None
    try:
        if db[COLL_USERS].find_one({"_id": user_id}):
            return "exists"
        now = _now()
        doc = {"_id": user_id, "email": email, "name": name or user_id,
               "role": role, "auth_provider": auth_provider,
               "created_at": now, "last_seen": now}
        db[COLL_USERS].insert_one(doc)
        return doc
    except Exception:
        return None


def ensure_default_admin() -> None:
    """Make sure an 'admin' user exists (maps to the shared Basic-auth login)."""
    try:
        if get_user("admin") is None:
            create_user("admin", name="Administrator", role="admin",
                        auth_provider="basic")
    except Exception:
        pass


def set_user_role(user_id: str, role: str, actor: str = "admin") -> bool:
    """v3.0.133 (RBAC): set a user's role (admin|member). Upserts so an admin can
    promote a user who hasn't logged in yet. Returns True on success."""
    if role not in ("admin", "member"):
        return False
    db = _get_db()
    if db is None:
        return False
    try:
        db[COLL_USERS].update_one(
            {"_id": user_id},
            {"$set": {"role": role}, "$setOnInsert": {"created_at": _now(),
             "name": user_id, "auth_provider": "unknown", "last_seen": _now()}},
            upsert=True)
        try:
            db[COLL_AUDIT].insert_one({"ts": _now(), "actor": actor,
                "event": "user.set_role", "target": {"kind": "user", "id": user_id},
                "after": {"role": role}})
        except Exception:
            pass
        return True
    except Exception:
        return False


def set_user_cluster_access(user_id: str, allow: bool, actor: str = "admin") -> bool:
    """v3.0.133 (RBAC): grant/revoke a member's permission to launch cluster
    (GPU) jobs. Admins always have access regardless of this flag."""
    db = _get_db()
    if db is None:
        return False
    try:
        db[COLL_USERS].update_one(
            {"_id": user_id},
            {"$set": {"can_use_cluster": bool(allow)},
             "$setOnInsert": {"created_at": _now(), "name": user_id,
             "role": "member", "auth_provider": "unknown", "last_seen": _now()}},
            upsert=True)
        try:
            db[COLL_AUDIT].insert_one({"ts": _now(), "actor": actor,
                "event": "user.cluster_access", "target": {"kind": "user", "id": user_id},
                "after": {"can_use_cluster": bool(allow)}})
        except Exception:
            pass
        return True
    except Exception:
        return False


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
