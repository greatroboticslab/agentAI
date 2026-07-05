"""
Atomic registry I/O for parallel Job-T (training) + Job-D (data) coordination.

Pattern: temp-write-then-rename (Lustre/POSIX guarantees atomic rename).
Reader never blocks; if a partial write is encountered, retry after backoff.
This is the same approach git, etcd, and CockroachDB use for crash-safe state.
"""

import contextlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

try:
    import fcntl  # POSIX only (cluster is Linux, dev is macOS — both have it)
    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - Windows fallback
    _HAVE_FCNTL = False


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


@contextlib.contextmanager
def registry_lock(path, timeout=120.0, poll=0.25):
    """Advisory exclusive lock for read-modify-write on a shared JSON file.

    Holds an `fcntl.flock(LOCK_EX)` on a `<path>.lock` sidecar. This gives
    CORRECT mutual exclusion between processes ON THE SAME NODE. On Lustre
    (/ocean) flock is honored across nodes only when the filesystem is mounted
    with flock support — Bridges-2 /ocean generally is, but callers must NOT
    treat cross-node exclusion as guaranteed. That is why every writer pairs
    this lock with `update_registry()` (re-read latest, then `atomic_write_json`):
    if the lock is ever not honored, the worst case degrades to a small
    last-writer-wins race, NEVER a torn/corrupt file or a wiped registry.

    On timeout it proceeds WITHOUT the lock (logging is the caller's job) rather
    than blocking a training/harvest job forever — again safe because the write
    itself is atomic.
    """
    if not _HAVE_FCNTL:
        yield False
        return
    lock_path = str(path) + ".lock"
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    f = open(lock_path, "w")
    held = False
    start = time.time()
    try:
        while True:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = True
                break
            except OSError:
                if time.time() - start > timeout:
                    break  # give up waiting; atomic write still prevents corruption
                time.sleep(poll)
        yield held
    finally:
        if held:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        f.close()


def update_registry(path, mutate_fn, default=None):
    """Safe locked read-modify-write for the shared registry.

    Re-reads the LATEST on-disk JSON *under the lock*, applies `mutate_fn(reg)`
    in place (or uses its return value if it returns a dict), then atomically
    writes it back. This is the correct replacement for the widespread
    load-once-at-start / save-whole-dict-hours-later pattern that let a training
    job clobber everything a concurrent harvest job wrote in the interim.

    Returns the written registry dict.
    """
    if default is None:
        default = {"datasets": {}, "discovered": []}
    with registry_lock(path):
        reg = safe_read_json(path)
        if reg is None:
            reg = default
        result = mutate_fn(reg)
        if result is not None:
            reg = result
        atomic_write_json(path, reg)
        return reg


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
