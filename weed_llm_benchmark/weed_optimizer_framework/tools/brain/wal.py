"""Crash-safe write-ahead helper for single-writer state files (WP5, v3.25.x).

Why this exists
----------------
A SLURM walltime kill can land the writer at any instruction boundary — the
documented case is the 2026-08-29 double TIMEOUT (jobs 44727703 / 44767709),
which died between deciding what the ledger should say and writing it. A plain
`open(path, "w")` is not atomic: a kill mid-write leaves a torn file, and one
that lands right after the write but before the caller records "done" leaves
no evidence that anything happened at all. Neither failure is rare enough to
ignore in a process that is deliberately walltime-bound.

Two primitives, one guarantee each:

* `atomic_write(path, data)` makes ONE replacement atomic: write to
  `<path>.tmp.<pid>` in the same directory, fsync the file, `os.replace` it
  into place, fsync the directory so the rename itself survives a crash, not
  just the bytes. A reader never observes a partial write — only the old
  content or the fully-written new content.
* `with_wal(path, mutate)` adds a durable record of INTENT around that
  replacement, because a single atomic write only protects the write itself,
  not the gap between deciding the next state and calling `atomic_write` for
  it. The intent (the fully-computed next state, itself installed via
  `atomic_write`) is written to `<path>.wal` before `path` is touched, and
  removed only after `path` matches it. A kill between "decided" and
  "written" — the exact gap the 2026-08-29 jobs died in — leaves the intent
  file behind; `recover(path)` (called automatically at the start of the next
  `with_wal`, and available for a caller to invoke directly on open) resolves
  it: reapply if `path` does not yet match, discard if it already does.

This module holds no opinion about what the bytes mean. It is stdlib-only so
it imports on the always-on lab server, inside a SLURM job body and on a
laptop. Single-writer discipline (only one process may call `with_wal` for a
given `path` at a time) is the CALLER's responsibility — `corrections.py`
enforces it with its own advisory lock before it ever calls in here; this
module only makes each individual replacement atomic and each mutation
recoverable, it does not arbitrate between concurrent writers.
"""
import os
import pathlib


def _fsync_dir(dirpath):
    """Best-effort fsync of a directory, so a rename is durable, not just the
    bytes of the file it points at. Some platforms have no directory file
    descriptor; failing here does not undo the rename that already happened.
    """
    try:
        target = str(dirpath) or "."
        fd = os.open(target, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _as_bytes(data):
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode("utf-8")
    raise TypeError("wal data must be bytes or str, got %r" % type(data))


def atomic_write(path, data):
    """Atomically replace the content of `path` with `data` (bytes or str).

    Returns the number of bytes written. Raises on a genuine I/O failure —
    this function's job is to make the write atomic, not to hide that it
    failed; a caller that must never raise (e.g. a worker loop) wraps this
    itself, the way `trace.append` wraps its own I/O.
    """
    p = pathlib.Path(str(path))
    if str(p.parent):
        p.parent.mkdir(parents=True, exist_ok=True)
    body = _as_bytes(data)
    tmp = str(p) + ".tmp.%d" % os.getpid()
    with open(tmp, "wb") as f:
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, str(p))
    _fsync_dir(p.parent)
    return len(body)


def _wal_path(path):
    return str(path) + ".wal"


def _discard_intent(path):
    wal_path = _wal_path(path)
    try:
        os.remove(wal_path)
    except OSError:
        pass
    _fsync_dir(pathlib.Path(str(path)).parent)


def recover(path):
    """Resolve a `.wal` intent left next to `path` by an interrupted `with_wal`.

    Returns True if an intent was found (and resolved either way), False if
    there was nothing pending. Because the intent file is itself written with
    `atomic_write`, "found" always means a complete, well-formed next state —
    there is no partially-written intent to reason about, only whether `path`
    already reflects it:

      * `path` already equals the intent -> the apply finished before the
        kill and only the final clear step was missed; discard the stale
        intent and leave `path` untouched.
      * `path` differs from the intent -> the kill landed before or during
        the apply, so `path` still holds the previous (complete, since it too
        was last written by `atomic_write`) state; reapply the intent to
        `path`, then discard it.

    Idempotent: a second call with nothing pending, or a second call right
    after the first resolved one, is a no-op.
    """
    wal_path = _wal_path(path)
    if not os.path.exists(wal_path):
        return False
    try:
        with open(wal_path, "rb") as f:
            intent = f.read()
    except OSError:
        return False
    current = None
    if os.path.exists(str(path)):
        try:
            with open(str(path), "rb") as f:
                current = f.read()
        except OSError:
            current = None
    if current != intent:
        atomic_write(path, intent)
    _discard_intent(path)
    return True


def with_wal(path, mutate):
    """Apply `mutate(current_bytes_or_None) -> new_bytes_or_None` to `path`,
    with a durable intent record covering the gap between deciding the next
    state and writing it.

    Any intent left by a previous, interrupted call is resolved first (see
    `recover`), so `mutate` always sees a state that is either the last fully
    completed write or is missing entirely — never a half-applied one. The
    return value of `mutate` is then recorded as intent (durably: this is
    itself an `atomic_write`, so the intent is either wholly absent or wholly
    present), applied to `path`, and only then discarded.

    `mutate` returning None means "nothing to write" (the caller decided,
    while composing the next state, to refuse); no intent is recorded and
    `path` is left untouched. Returns the bytes written, or None if `mutate`
    declined.
    """
    recover(path)
    current = None
    if os.path.exists(str(path)):
        with open(str(path), "rb") as f:
            current = f.read()
    new_data = mutate(current)
    if new_data is None:
        return None
    new_bytes = _as_bytes(new_data)
    atomic_write(_wal_path(path), new_bytes)
    atomic_write(path, new_bytes)
    _discard_intent(path)
    return new_bytes
