"""Shared read/write for results/framework/class_topic_overrides.json.

Three callers:
  1. dashboard_server (serves GET/POST /api/class_topic, reads on /classes render)
  2. dataset_discovery (writes after Brain LLM classifies new harvest classes)
  3. topic_classifier (uses store + LLM together)

Single source of truth — avoids two modules each defining their own io path.
"""
from __future__ import annotations
import json
import os
import threading
from pathlib import Path
from typing import Optional

# Where the override file lives. Overridable via env for tests.
DEFAULT_FILE = os.environ.get(
    "CLASS_TOPIC_OVERRIDES_FILE",
    "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/"
    "results/framework/class_topic_overrides.json",
)

VALID_TOPICS = ("cwd12", "weed", "disease", "pest", "crop", "other")

_lock = threading.Lock()
_cache: dict = {"mtime": 0.0, "data": {}}


def load_overrides(file: Optional[str] = None) -> dict:
    """Return {class_name: topic} from the overrides file. mtime-cached."""
    fp = Path(file or DEFAULT_FILE)
    if not fp.exists():
        return {}
    try:
        mt = fp.stat().st_mtime
    except Exception:
        return {}
    with _lock:
        if _cache["mtime"] == mt:
            return dict(_cache["data"])
        try:
            with open(fp) as f:
                data = json.load(f) or {}
        except Exception:
            return dict(_cache["data"])
        _cache["mtime"] = mt
        _cache["data"] = data
        return dict(data)


def save_override(cls: str, topic: str, file: Optional[str] = None) -> bool:
    """Persist a single class→topic. topic='_clear_' removes the entry.
    Returns True on success."""
    if topic == "_clear_":
        pass
    elif topic not in VALID_TOPICS:
        return False
    fp = Path(file or DEFAULT_FILE)
    with _lock:
        data = dict(_cache["data"])
        # Fresh-read from disk so concurrent writers don't lose changes
        if fp.exists():
            try:
                with open(fp) as f:
                    data = json.load(f) or {}
            except Exception:
                pass
        if topic == "_clear_":
            data.pop(cls, None)
        else:
            data[cls] = topic
        try:
            fp.parent.mkdir(parents=True, exist_ok=True)
            tmp = fp.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.replace(tmp, fp)
            _cache["mtime"] = 0.0  # force re-read next time
            return True
        except Exception:
            return False


def save_overrides_bulk(updates: dict, file: Optional[str] = None) -> int:
    """Batch-save many class→topic pairs in one atomic write. Returns n_written.
    Skips entries with invalid topics (no exception)."""
    fp = Path(file or DEFAULT_FILE)
    valid_updates = {
        c: t for c, t in updates.items()
        if t in VALID_TOPICS or t == "_clear_"
    }
    if not valid_updates:
        return 0
    with _lock:
        data = {}
        if fp.exists():
            try:
                with open(fp) as f:
                    data = json.load(f) or {}
            except Exception:
                pass
        for c, t in valid_updates.items():
            if t == "_clear_":
                data.pop(c, None)
            else:
                data[c] = t
        try:
            fp.parent.mkdir(parents=True, exist_ok=True)
            tmp = fp.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.replace(tmp, fp)
            _cache["mtime"] = 0.0
            return len(valid_updates)
        except Exception:
            return 0
