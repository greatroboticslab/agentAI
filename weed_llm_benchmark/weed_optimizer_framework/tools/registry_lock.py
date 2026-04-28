"""
Atomic registry I/O for parallel Job-T (training) + Job-D (data) coordination.

Pattern: temp-write-then-rename (Lustre/POSIX guarantees atomic rename).
Reader never blocks; if a partial write is encountered, retry after backoff.
This is the same approach git, etcd, and CockroachDB use for crash-safe state.
"""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path


def atomic_write_json(path, data):
    """Atomically replace the JSON at `path` with `data`. Crash-safe on Lustre.

    Caller MUST hold any application-level coordination they need (e.g., be
    the sole writer in their job role); this function only protects against
    partial writes / readers seeing torn state.
    """
    path = str(path)
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic on POSIX/Lustre
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def safe_read_json(path, retries=5, retry_sleep=0.2):
    """Read JSON, retrying on JSONDecodeError (writer might be mid-rename).

    Returns parsed dict or None if file truly doesn't exist / unreadable.
    """
    path = str(path)
    for attempt in range(retries):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            time.sleep(retry_sleep)
    return None


def snapshot_registry(src_path, snapshot_dir):
    """Job-T calls this at the start of each mini-round. Copies the live
    registry to a frozen snapshot file the trainer reads from for the
    duration of that round, so concurrent Job-D writes don't perturb
    the merge.

    Returns path to the snapshot.
    """
    os.makedirs(snapshot_dir, exist_ok=True)
    snap_path = os.path.join(snapshot_dir, f"registry_{int(time.time())}.json")
    data = safe_read_json(src_path) or {"datasets": {}, "discovered": []}
    atomic_write_json(snap_path, data)
    return snap_path


def diff_dataset_slugs(old_snapshot_path, new_snapshot_path):
    """Return dataset slugs added between two snapshots. Used by Job-T to
    decide whether the next mini-round needs to re-merge data.
    """
    old = safe_read_json(old_snapshot_path) or {"datasets": {}}
    new = safe_read_json(new_snapshot_path) or {"datasets": {}}
    old_slugs = set(old.get("datasets", {}).keys())
    new_slugs = set(new.get("datasets", {}).keys())
    return new_slugs - old_slugs
