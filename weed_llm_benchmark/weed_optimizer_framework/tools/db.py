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
import re
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
COLL_ROUNDS = "domain_rounds"     # v3.0.186 (P4): per-domain compounding-loop round provenance

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


# v3.0.179 (Phase 1) — PER-DOMAIN CONFIG. The platform is domain-general: every
# tunable that used to be a weed/CWD12 hardcode now lives in a per-project config,
# with these DEFAULTS reproducing today's behaviour EXACTLY (so nothing regresses).
# A new research field = a new config, not a code change.
DEFAULT_DOMAIN_CONFIG = {
    "taxonomy": [],            # class / species / label names for this field
    "harvest_queries": [],     # collector search terms
    "accept_vocab": [],        # words that make a harvested dataset relevant
    "thresholds": {            # dataset-quality knobs (today's hardcoded values)
        "dino_threshold": 0.45,
        "imbalance_high": 10, "imbalance_med": 3,
        "dup_frac": 0.10, "tiny_px": 64,
        "min_per_class": 10, "small_dataset": 100,
    },
    "reference_pool_policy": "uploaded_or_labeled",   # filter reference pool
    "roboflow_project": "",    # labeler target ("" → derived per domain)
    "modality": "image",
    "target_metric": "",       # "" → per-task default
    "model_routing": {},       # role -> model id overrides
    # v3.25.0 — the loop's step commands lived as three hardcoded sbatch
    # literals in round_scheduler._WEED_STEPS, so changing one parameter was a
    # code change and the 2026-08-29 walltime failures (jobs 44727703,
    # 44767709) could only be corrected by hand. Rendering them from per-domain
    # templates keeps the same commands (collect and filter render byte-identical
    # to those literals — tests/test_round_ledger.py asserts it) and gives a
    # second domain its own knobs with no code change.
    "steps": {
        "collect": ("sbatch --time={collect_time_h}:00:00 "
                    "--export=ALL,BRAIN_MAX_NEW={max_new} "
                    "run_v3_0_43_brain_harvest_oneshot.sh"),
        "filter": "sbatch run_s2_dino_scores.sh",
        "train": ("sbatch --array=1-1 --job-name=rndtrain --gres=gpu:h100-80:1 "
                  "--time={train_time_h}:00:00 --export=ALL,TIER={tier},"
                  "MIN_DINO_SCORE={min_dino_score},TRAIN_EPOCHS={epochs},"
                  "TRAIN_TIME_H={train_time_cap_h},IMGSZ={imgsz},"
                  "PATIENCE={patience},ITER_NAME={iter_name},"
                  "BRAIN_DOMAIN={domain},BRAIN_TRACE={trace} "
                  "run_m1_merged_seeds.sh"),
    },
    # train_time_cap_h is 0.9 × train_time_h: the trainer's own time cap ends the
    # run with a valid best.pt before SLURM kills the job. Its absence is why two
    # consecutive 12 h train steps ended TIMEOUT at epoch 24/60 and 16/60 with no
    # metric and no checkpoint.
    "round_params": {"collect_time_h": 10, "max_new": 3, "tier": "curated",
                     "min_dino_score": 0.50, "epochs": 60, "train_time_h": 12,
                     "train_time_cap_h": 10.8, "imgsz": 640, "patience": 20},
    # "scripted" = the deterministic loop, no model decides anything (the
    # baseline). An empty tier id means that tier is not wired, so a missing
    # deployment can never silently promote another model into a review role.
    "brain": {"policy": "scripted",
              "tiers": {"worker": "", "fast": "", "deep": "", "planner": ""},
              "review_timeout_min": 90, "periodic_audit_every": 4},
    "budget": {"su_envelope": 1500, "daily_cap": 120, "per_round_cap": 60},
    # Sealed 2× seed-std per recipe: a round-to-round delta under this value is
    # noise and must not be reported as an effect.
    "noise_floor": {"merged_curated": 0.005, "merged_raw": 0.009,
                    "cwd12_core": 0.006},
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def get_domain_config(domain_id: str) -> dict:
    """Effective config for a domain: DEFAULTS <- domain.config, back-filling
    harvest_queries / taxonomy / modality from the legacy domain fields so existing
    domains (incl. weed) work with zero migration."""
    import copy as _copy
    cfg = _copy.deepcopy(DEFAULT_DOMAIN_CONFIG)
    d = get_domain(domain_id) or {}
    cfg = _deep_merge(cfg, d.get("config") or {})
    # back-fill from legacy top-level fields when config didn't set them
    if not cfg["harvest_queries"] and d.get("harvest_queries"):
        cfg["harvest_queries"] = list(d["harvest_queries"])
    if not cfg["taxonomy"] and d.get("taxonomy"):
        cfg["taxonomy"] = list(d["taxonomy"])
    _mods = d.get("modality")
    if _mods:
        cfg["modality"] = _mods[0] if isinstance(_mods, list) and _mods else _mods
    return cfg


# v3.25.0 moved the loop's submit commands out of code and into
# DEFAULT_DOMAIN_CONFIG["steps"], which is patchable at runtime through the
# domain-config endpoint. Before that, changing what the unattended loop submits
# took a code edit and a deploy; now a rendered template reaches `bash -lc` on
# the shared cluster account. These two constants and validate_step_command()
# are the one choke point between a project config and that shell.
# Metacharacters that could chain, substitute or redirect. `$` is refused whole,
# not just `$(`: `${x}` and `$x` expand as well.
_STEP_CMD_METACHARS = ";&|`$()<>"
# A literal script name, no path separator, so a template can never point the
# loop outside the repository root.
_STEP_SCRIPT_RE = re.compile(r"run_[a-z0-9_.-]+\.sh")
_STEP_CMD_MAX = 4000
# The scripts the round scheduler may submit. This is CODE, not config: a
# per-domain `steps` template can choose parameters, never a new executable, and
# a project owner patching config over HTTP therefore cannot introduce one. A new
# research domain reuses these entry points with its own parameters; adding an
# entry point is a code change on purpose.
#
# Membership replaced an is_file() check on the repository root. The renderer runs
# on the always-on server while the scripts live in the cluster checkout, so that
# check refused every legitimate template wherever the two trees differ — and a
# refusal drops the command, which would have stopped the loop silently. Existence
# is verified where it is authoritative and loud: sbatch fails immediately on a
# missing script and the submission is recorded failed.
_STEP_ALLOWED_SCRIPTS = frozenset({
    "run_v3_0_43_brain_harvest_oneshot.sh",   # collect
    "run_s2_dino_scores.sh",                  # filter
    "run_m1_merged_seeds.sh",                 # train
    "run_train_generic.sh",                   # per-project training
    "run_eval_generic.sh",                    # per-project evaluation
})


def validate_step_command(cmd) -> tuple:
    """(ok, reason) for a RENDERED step command. Allow-list, not deny-list.

    THE choke point between a project config and a shell on the cluster: the
    round scheduler renders `steps` templates from a per-domain config that any
    project owner can patch over HTTP, then submits the result through
    `bash -lc` under the shared account. Nothing else stands between the two, so
    a command is submittable only when all of these hold:

      * it starts with "sbatch " — the loop submits batch jobs, nothing else;
      * it contains none of ; & | ` $ ( ) < > and no control character, so it
        cannot chain, substitute, redirect or span lines;
      * its last token is a literal run_*.sh from _STEP_ALLOWED_SCRIPTS, so a
        config patch can choose parameters but never a new executable.

    Refusing is always safe: the caller drops the command instead of submitting
    it, which stops a round rather than running an unreviewed one.
    """
    if not isinstance(cmd, str) or not cmd.strip():
        return False, "empty command"
    if len(cmd) > _STEP_CMD_MAX:
        return False, "command is longer than %d characters" % _STEP_CMD_MAX
    if not cmd.startswith("sbatch "):
        return False, "must start with 'sbatch '"
    bad = sorted({c for c in cmd if c in _STEP_CMD_METACHARS or ord(c) < 32})
    if bad:
        return False, ("contains shell metacharacter(s) %s"
                       % " ".join(repr(c) for c in bad))
    script = cmd.split()[-1]
    if not _STEP_SCRIPT_RE.fullmatch(script):
        return False, ("must end in a run_*.sh script name, not %r" % script[:80])
    if script not in _STEP_ALLOWED_SCRIPTS:
        return False, ("script %r is not one the scheduler may submit" % script[:80])
    return True, ""


class _ProbeFields(dict):
    """Render fields for a not-yet-submitted template: the config's own
    round_params, with a harmless literal for anything the scheduler supplies
    per submission. A field standing in for the script name therefore fails
    validate_step_command's last-token check, which is the point — the script
    the loop runs must be literal in the template, not chosen at submit time."""

    def __missing__(self, key):
        return "0"


def _validate_steps_patch(effective_cfg: dict, patch_steps: dict) -> tuple:
    """(ok, reason) for the `steps` block of a config patch, checked as it would
    render. Structure only: the values a submission fills in are re-checked at
    render time, where their real values are known."""
    if not isinstance(patch_steps, dict):
        return False, "steps must be an object"
    for step, tpl in patch_steps.items():
        if not isinstance(tpl, str) or not tpl.strip():
            return False, "step %r has no command template" % (step,)
        fields = _ProbeFields((effective_cfg or {}).get("round_params") or {})
        try:
            rendered = tpl.format_map(fields)
        except Exception as e:
            return False, "step %r does not render (%s)" % (step, str(e)[:80])
        ok, why = validate_step_command(rendered)
        if not ok:
            return False, "step %r: %s" % (step, why)
    return True, ""


def set_domain_config(domain_id: str, patch: dict, actor: str = "user") -> Optional[dict]:
    """Deep-merge patch into a domain's config, persist, audit. Returns the new
    effective config, or None if Mongo down / domain missing."""
    db = _get_db()
    if db is None:
        return None
    try:
        d = db[COLL_DOMAINS].find_one({"_id": domain_id})
        if not d:
            return None
        new_cfg = _deep_merge(d.get("config") or {}, patch or {})
        # A `steps` patch is a remote-execution surface: the unattended loop
        # renders it and runs it through `bash -lc` on the shared account. An
        # unsafe template is refused whole and recorded — never half-stored, and
        # never stored to be caught later at submit time (v3.25.0).
        if "steps" in (patch or {}):
            import copy as _copy
            _probe = _deep_merge(_copy.deepcopy(DEFAULT_DOMAIN_CONFIG), new_cfg)
            _ok, _why = _validate_steps_patch(_probe, (patch or {}).get("steps"))
            if not _ok:
                try:
                    db[COLL_AUDIT].insert_one({"ts": _now(), "actor": actor,
                        "event": "domain.config.refused",
                        "target": {"kind": "domain", "id": domain_id},
                        "after": {"reason": _why, "patch": patch}})
                except Exception:
                    pass
                return None
        db[COLL_DOMAINS].update_one({"_id": domain_id}, {"$set": {"config": new_cfg}})
        try:
            db[COLL_AUDIT].insert_one({"ts": _now(), "actor": actor,
                "event": "domain.config", "target": {"kind": "domain", "id": domain_id},
                "after": patch})
        except Exception:
            pass
        return get_domain_config(domain_id)
    except Exception:
        return None


# ===========================================================================
# v3.0.186 (P4) — per-domain "round" provenance for the closed compounding loop.
# A round is one pass of collect -> filter -> label -> train -> eval; each step
# records WHAT ran, WHEN, by WHOM, with what result. eval metrics land on the
# round so the next collect can be biased by them (the compounding feedback).
# This is a GENERAL, per-domain layer (any modality/field) — separate from the
# weed-registry harvest_round tags (which track which datasets, not which steps).
# ===========================================================================
ROUND_STEPS = ["collect", "filter", "label", "train", "eval"]
_ROUND_STATUSES = {"pending", "running", "done", "failed", "skipped"}


def _now_iso() -> str:
    """ISO-8601 UTC string — round docs are returned raw via JSONResponse, so their
    timestamps must be JSON-serializable (unlike _now()'s datetime)."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _round_id(domain_id: str, n: int) -> str:
    return f"{domain_id}#{int(n)}"


# v3.25.0 — how many superseded head entries a step keeps. Bounded because the
# round doc is returned whole to the project page on every poll.
_MAX_STEP_ATTEMPTS = 20
# Actors whose entry is a supervision record: a diagnosis, a correction or a
# human decision. Never overwritten by the automation below.
_OWNER_ACTOR_PREFIXES = ("tier1:", "tier2:", "human:")
# Actors that run unattended every tick.
_AUTOMATION_ACTORS = ("round-scheduler",)
_AUTOMATION_ACTOR_PREFIXES = ("tier0:",)


def _round_step_entry(status: str, detail=None, job=None, actor: str = "user",
                      now: str = "", params=None, decided_by=None,
                      review=None, su=None) -> dict:
    """Pure builder for one step's provenance entry (unit-testable, no Mongo).

    v3.25.0 adds the supervision fields, all optional and written only when
    supplied so a pre-v3.25.0 caller still produces the old three-key entry:
    `params` (what the step was actually rendered with), `decided_by`
    (default|rule|advisory|tier1|human), `review`
    ({status, review_id, queued_at}) and `su`."""
    st = status if status in _ROUND_STATUSES else "pending"
    e = {"status": st, "actor": actor, "at": now}
    if job:
        e["job"] = str(job)[:120]
    if detail is not None:
        e["detail"] = detail
    if isinstance(params, dict) and params:
        e["params"] = dict(params)
    if decided_by:
        e["decided_by"] = str(decided_by)[:40]
    if isinstance(review, dict) and review:
        e["review"] = dict(review)
    if su is not None:
        try:
            e["su"] = float(su)
        except (TypeError, ValueError):
            pass
    return e


def _is_owner_actor(actor) -> bool:
    return str(actor or "").startswith(_OWNER_ACTOR_PREFIXES)


def _is_automation_actor(actor) -> bool:
    a = str(actor or "")
    return a in _AUTOMATION_ACTORS or a.startswith(_AUTOMATION_ACTOR_PREFIXES)


def _as_attempt(entry: dict) -> dict:
    """One history row: a head entry without its own history (no nesting)."""
    return {k: v for k, v in entry.items() if k != "attempts"}


def merge_step_entry(existing, incoming: dict, incoming_actor: str = "") -> dict:
    """Merge a new step entry onto the one already on the round doc. Pure.

    Two properties the loop needs that a plain $set cannot give (v3.25.0):

      * history — the head used to be replaced outright, so a retry's
        "running" erased the "failed" it was retrying. The 2026-08-29 double
        TIMEOUT therefore left one step entry on the ledger for two burnt 12 h
        jobs. The previous head is now pushed onto `attempts` (oldest first,
        capped at _MAX_STEP_ATTEMPTS) on every write.
      * ownership — a diagnosis or correction written by a supervisor
        (`tier1:`/`tier2:`) or a person (`human:`) must survive the next
        scheduler tick. Such a head stays in place and the automated write is
        recorded as an attempt instead, so nothing is lost either way.

    `existing` may be None (first write) or a legacy entry with no `attempts`.
    """
    prev = existing if isinstance(existing, dict) else None
    attempts = []
    if prev and isinstance(prev.get("attempts"), list):
        attempts = [a for a in prev["attempts"] if isinstance(a, dict)]
    inc = dict(incoming or {})
    actor = incoming_actor or inc.get("actor") or ""
    if prev and _is_owner_actor(prev.get("actor")) and _is_automation_actor(actor):
        head = _as_attempt(prev)
        attempts = attempts + [_as_attempt(inc)]
    else:
        head = _as_attempt(inc)
        if prev:
            attempts = attempts + [_as_attempt(prev)]
    head["attempts"] = attempts[-_MAX_STEP_ATTEMPTS:]
    return head


def render_step_command(domain_cfg: dict, step: str, **fields) -> str:
    """Render one step's submit command from the domain config's template.

    v3.25.0: the scheduler held these as literals, so a parameter change was a
    code change and a correction could not reach a running campaign. Rendering
    from `steps` + `round_params` (plus per-submission `fields` such as
    iter_name / domain / trace) keeps the defaults byte-identical while letting
    a config or a correction change one field. A missing placeholder raises
    KeyError and a command that fails validate_step_command raises ValueError —
    a half-rendered or unsafe sbatch line must never be submitted."""
    steps = (domain_cfg or {}).get("steps") or {}
    tpl = steps.get(step)
    if not isinstance(tpl, str) or not tpl:
        raise KeyError("no command template for step %r in this domain config" % (step,))
    vals = dict((domain_cfg or {}).get("round_params") or {})
    vals.update(fields)
    try:
        cmd = tpl.format(**vals)
    except KeyError as e:
        missing = e.args[0] if e.args else "?"
        raise KeyError("step %r command template needs field %r; have %s"
                       % (step, missing, sorted(vals))) from None
    # Checked here rather than at each call site: this is the only place a
    # config value becomes a command line, and the values substituted above come
    # from the same patchable config (v3.25.0).
    ok, why = validate_step_command(cmd)
    if not ok:
        raise ValueError("step %r command is not submittable: %s" % (step, why))
    return cmd


def start_round(domain_id: str, actor: str = "user") -> Optional[dict]:
    """Open the next round for a domain (max existing round_num + 1, else 1).
    Returns the new round doc, or None if Mongo is down."""
    db = _get_db()
    if db is None:
        return None
    try:
        last = db[COLL_ROUNDS].find_one({"domain": domain_id},
                                        sort=[("round_num", -1)])
        n = int((last or {}).get("round_num", 0)) + 1
        now = _now_iso()
        doc = {"_id": _round_id(domain_id, n), "domain": domain_id,
               "round_num": n, "created_at": now, "updated_at": now,
               "actor": actor, "status": "open", "steps": {}, "metrics": {}}
        db[COLL_ROUNDS].insert_one(doc)
        try:
            db[COLL_AUDIT].insert_one({"ts": _now(), "actor": actor,
                "event": "round.start", "target": {"kind": "domain", "id": domain_id},
                "after": {"round_num": n}})
        except Exception:
            pass
        return doc
    except Exception:
        return None


def record_round_step(domain_id: str, step: str, status: str, detail=None,
                      job=None, actor: str = "user", round_num=None,
                      metrics=None, params=None, decided_by=None,
                      review=None, su=None) -> Optional[dict]:
    """Record a pipeline step's provenance on a round (the current open one unless
    round_num is given; auto-opens round 1 if none exists). `metrics` (from eval)
    merge onto round.metrics — the compounding feedback the next collect can read.
    Returns the updated round doc, or None if Mongo down / bad step.

    v3.25.0: `params`, `decided_by`, `review` and `su` are optional supervision
    fields (see _round_step_entry), and the write goes through merge_step_entry
    so the previous head survives in `attempts` and a tier-1/tier-2/human entry
    is not overwritten by the scheduler. Old docs without `attempts` are read
    unchanged."""
    if step not in ROUND_STEPS:
        return None
    db = _get_db()
    if db is None:
        return None
    try:
        if round_num is None:
            cur = db[COLL_ROUNDS].find_one({"domain": domain_id},
                                           sort=[("round_num", -1)])
            if not cur:
                cur = start_round(domain_id, actor=actor)
                if not cur:
                    return None
            round_num = cur["round_num"]
        now = _now_iso()
        rid = _round_id(domain_id, round_num)
        entry = _round_step_entry(status, detail, job, actor, now,
                                  params=params, decided_by=decided_by,
                                  review=review, su=su)
        # Read-modify-write (the step name is validated against ROUND_STEPS
        # above, so the projection key is not caller-controlled).
        try:
            _steps = (db[COLL_ROUNDS].find_one({"_id": rid},
                                               {f"steps.{step}": 1}) or {}).get("steps")
            head = (_steps or {}).get(step)
        except Exception:
            head = None
        sets = {f"steps.{step}": merge_step_entry(head, entry, actor),
                "updated_at": now}
        if isinstance(metrics, dict) and metrics:
            for k, v in metrics.items():
                sets[f"metrics.{k}"] = v
        db[COLL_ROUNDS].update_one({"_id": rid}, {"$set": sets}, upsert=True)
        try:
            db[COLL_AUDIT].insert_one({"ts": _now(), "actor": actor,
                "event": "round.step", "target": {"kind": "domain", "id": domain_id},
                "after": {"round_num": round_num, "step": step, "status": status}})
        except Exception:
            pass
        return db[COLL_ROUNDS].find_one({"_id": rid})
    except Exception:
        return None


def get_rounds(domain_id: str, limit: int = 25) -> list:
    """All rounds for a domain, newest first. [] if Mongo down / none."""
    db = _get_db()
    if db is None:
        return []
    try:
        return list(db[COLL_ROUNDS].find({"domain": domain_id},
                                         sort=[("round_num", -1)]).limit(int(limit)))
    except Exception:
        return []


def get_current_round(domain_id: str) -> Optional[dict]:
    """The highest-numbered round doc for a domain, or None."""
    db = _get_db()
    if db is None:
        return None
    try:
        return db[COLL_ROUNDS].find_one({"domain": domain_id},
                                        sort=[("round_num", -1)])
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
            try:  # v3.0.186.2: cascade the project's round provenance
                db[COLL_ROUNDS].delete_many({"domain": domain_id})
            except Exception:
                pass
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


def add_project_agent(domain_id: str, atype: str, name: str = "",
                      actor: str = "user", model: str = "auto") -> Optional[dict]:
    """v3.0.146: add an AGENT component to a PROJECT (domain). A project holds
    0..N agents, freely composed (collector/filter/labeler/trainer/evaluator/
    custom). Returns the new agent dict, or None."""
    import secrets as _s
    db = _get_db()
    if db is None:
        return None
    agent = {"id": _s.token_hex(4), "type": atype, "name": name or atype,
             "created_at": _now().isoformat(), "status": "idle",
             "config": {"model": model or "auto"}}
    try:
        r = db[COLL_DOMAINS].update_one({"_id": domain_id}, {"$push": {"agents": agent}})
        if r.matched_count == 0:
            return None
        try:
            db[COLL_AUDIT].insert_one({"ts": _now(), "actor": actor,
                "event": "project.agent_add", "target": {"kind": "domain", "id": domain_id},
                "after": {"agent": agent["type"]}})
        except Exception:
            pass
        return agent
    except Exception:
        return None


def remove_project_agent(domain_id: str, agent_id: str, actor: str = "user") -> bool:
    db = _get_db()
    if db is None:
        return False
    try:
        r = db[COLL_DOMAINS].update_one({"_id": domain_id},
                                        {"$pull": {"agents": {"id": agent_id}}})
        return bool(r.modified_count)
    except Exception:
        return False


def create_domain(domain_id: str, display_name: str, taxonomy=None,
                  target_metric: str = None, harvest_queries=None,
                  n_subagents: int = 2, status: str = "created",
                  task: str = "detection", modality=None,
                  model: str = "auto", owner: str = "",
                  research_field: str = "") -> Optional[dict]:
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
            "research_field": research_field or "",
            "agents": [],
            "created_at": _now().isoformat(),
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


def list_audit(limit: int = 100, actor: str = "") -> list:
    """Recent audit-trail events, newest first. Each: {ts(iso), actor, event,
    target, after, before, reason}. Mongo-only; [] if Mongo is down."""
    db = _get_db()
    if db is None:
        return []
    q = {}
    if actor:
        q["actor"] = actor
    try:
        rows = list(db[COLL_AUDIT].find(q).sort("ts", -1).limit(max(1, min(int(limit), 500))))
    except Exception:
        return []
    out = []
    for r in rows:
        r.pop("_id", None)
        ts = r.get("ts")
        try:
            r["ts"] = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        except Exception:
            r["ts"] = str(ts)
        out.append(r)
    return out


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
