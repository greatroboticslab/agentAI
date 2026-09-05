"""Append-only, hash-chained artifact trace (v3.25.0).

Why this exists
---------------
On 2026-08-29 two train jobs (44727703, 44767709) ran into the 12 h walltime and
were killed. Everything the jobs knew died with them: the metric writer only ran
at the end, so the ledger recorded `TIMEOUT` and nothing else. After the fact
there was no way to tell a walltime-bound run (epochs too many for a pool that
grew every round) from an actual training defect, and no way to see it coming
while the job was still alive.

The fix is a trace the worker writes *as it goes*: one JSON line per epoch, per
harvest candidate, per decision, appended to a file on /ocean from the first
epoch. A step that dies still leaves its own evidence behind, and a step that is
still running can be projected against its walltime on the next scheduler tick.

Two properties make the file usable as evidence rather than as a log:

* **Append-only, one line per write.** Writers are SLURM jobs on shared nodes and
  a scheduler thread on the lab; none of them coordinate. Every line is written
  with a single `write()` to a handle opened in "a" mode (O_APPEND), so
  concurrent writers interleave whole lines instead of corrupting each other.
  Nothing here ever rewrites or truncates an existing file. The chain on top of
  that assumes one writer per file, which the naming convention gives
  (`<round>_<step>_<jobid>.jsonl`): two processes appending to the same file in
  the same instant can read the same predecessor, and `verify()` reports the
  fork rather than hiding it.
* **Hash chain.** Each record carries `sha` (sha256 over its own canonical JSON,
  minus that field) and `sha_prev` (the `sha` of the record before it). A line
  edited or removed after the fact breaks the chain at that point, which
  `verify()` reports. All jobs run under one account, so this detects tampering
  and truncation; it cannot prevent them.

Reading the previous `sha` on append seeks to the end and reads the last 4 KB
rather than parsing the file, because the caller is an Ultralytics callback that
runs once per epoch on a Lustre filesystem.

No function here raises. A trace is an observation of a run, never a reason to
kill one: `append()` returns an empty dict when it cannot write.

Record fields in use: `ts`, `kind` (epoch, candidate, decision, report, start,
end), `domain`, `round`, `step`, `job_id`, plus the fields of that kind.
"""
import hashlib
import json
import os
import pathlib
import time

# Enough for several records of any shape this writes, small enough that the
# read costs one Lustre round trip per epoch.
TAIL_BYTES = 4096


def _canonical(record):
    """Canonical JSON for hashing: sorted keys, no padding, ASCII-escaped."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def _sha_of(record):
    """sha256 of the canonical JSON of `record` (which must carry no "sha")."""
    return hashlib.sha256(_canonical(record).encode("utf-8")).hexdigest()


def _last_sha_in(blob):
    """Last `sha` in a chunk of trace bytes, scanning backwards.

    Unparseable lines are skipped rather than treated as the end of the chain:
    the tail of a file whose writer was killed mid-write is a half line, and the
    next record must still link to the last record that was written whole.
    """
    try:
        text = blob.decode("utf-8", "replace")
    except Exception:
        return ""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if isinstance(rec, dict) and isinstance(rec.get("sha"), str) and rec["sha"]:
            return rec["sha"]
    return ""


def _chain_tail(path):
    """State of the end of `path` as (last sha, file ends with a newline).

    Returns ("", True) for a file that does not exist or is empty. The newline
    flag matters because a writer killed mid-write leaves a line with no
    terminator: appending straight onto it would glue the next record to the
    fragment and lose that record too, turning one torn line into an unbounded
    hole.
    """
    try:
        size = os.path.getsize(str(path))
    except OSError:
        return "", True
    if size <= 0:
        return "", True
    try:
        with open(str(path), "rb") as f:
            if size > TAIL_BYTES:
                f.seek(size - TAIL_BYTES)
                raw = f.read(TAIL_BYTES)
            else:
                raw = f.read()
        ends_nl = raw.endswith(b"\n")
        chunk = raw
        if size > TAIL_BYTES:
            # The window almost certainly opens mid-line; that fragment is not a
            # record and must not be parsed as one.
            cut = chunk.find(b"\n")
            chunk = chunk[cut + 1:] if cut >= 0 else b""
        sha = _last_sha_in(chunk)
        if sha or size <= TAIL_BYTES:
            return sha, ends_nl
        # A record longer than the window (a fat report line). Pay for the full
        # read rather than silently starting a second chain in the same file.
        with open(str(path), "rb") as f:
            return _last_sha_in(f.read()), ends_nl
    except OSError:
        return "", True


def append(path, record):
    """Append one record to the trace at `path`; return the record as written.

    Fills in `ts` when the caller did not, links `sha_prev` to the last record in
    the file (empty string for a new file) and stamps `sha`. Parent directories
    are created. Returns {} on any failure — callers are worker loops and job
    bodies where a trace write must never be the thing that ends the run.
    """
    try:
        if not isinstance(record, dict):
            return {}
        rec = dict(record)
        rec.pop("sha", None)
        if "ts" not in rec:
            rec["ts"] = time.time()
        p = pathlib.Path(str(path))
        if str(p.parent):
            p.parent.mkdir(parents=True, exist_ok=True)
        prev_sha, ends_nl = _chain_tail(p)
        rec["sha_prev"] = prev_sha
        # Round-trip through JSON before hashing so the digest is taken over
        # exactly what a reader will parse back (paths, numpy scalars and other
        # objects go through `default=str` on the way out).
        body = json.loads(json.dumps(rec, default=str))
        body["sha"] = _sha_of({k: v for k, v in body.items() if k != "sha"})
        line = _canonical(body) + "\n"
        if not ends_nl:
            # Keep the unterminated fragment on its own (skipped) line.
            line = "\n" + line
        with open(str(p), "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
        return body
    except Exception:
        return {}


def read(path, limit=None):
    """Records in `path`, oldest first; the last `limit` when given.

    Lines that do not parse are skipped, so the half line left by a killed
    writer hides neither the records before it nor those a later job appends.
    """
    out = []
    try:
        with open(str(path), "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except Exception:
        return []
    if limit is None:
        return out
    try:
        n = int(limit)
    except (TypeError, ValueError):
        return out
    if n <= 0:
        return []
    return out[-n:]


def latest(path, kind=None):
    """Most recent record in `path`, or the most recent of `kind`; None if none."""
    for rec in reversed(read(path)):
        if kind is None or rec.get("kind") == kind:
            return rec
    return None


def verify(path):
    """Check the hash chain. Returns {ok, records, broken_at}.

    `records` counts the records verified whole; when `ok` is False that is the
    number before the break and `broken_at` is the 1-based index (among parsed
    records) of the first record whose own `sha` or whose link to its
    predecessor does not hold up. A missing or empty file is `ok` with 0
    records — a step that wrote no trace is a different fault from a trace that
    was edited, and the caller tells them apart by `records`.

    Unparseable lines are skipped, exactly as `append()` skips them when it
    links a new record: a job killed at its walltime leaves a half-written last
    line, and reporting every killed job as a broken chain would make the check
    useless. Detection does not rest on line parseability — it rests on the
    links, so a record that is edited, replaced with debris or deleted breaks
    the `sha_prev` of the record that follows it. The one case this cannot see
    is the loss of the final record(s), which is indistinguishable from a writer
    that was killed before it got there.
    """
    out = {"ok": True, "records": 0, "broken_at": None}
    try:
        if not os.path.exists(str(path)):
            return out
        with open(str(path), "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except Exception:
        return {"ok": False, "records": 0, "broken_at": None}
    prev = ""
    idx = 0
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        idx += 1
        sha = rec.get("sha")
        body = {k: v for k, v in rec.items() if k != "sha"}
        if not isinstance(sha, str) or sha != _sha_of(body):
            return {"ok": False, "records": idx - 1, "broken_at": idx}
        if rec.get("sha_prev", "") != prev:
            return {"ok": False, "records": idx - 1, "broken_at": idx}
        prev = sha
    out["records"] = idx
    return out
