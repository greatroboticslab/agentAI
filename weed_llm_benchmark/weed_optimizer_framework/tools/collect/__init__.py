"""The targeted collector of the continuous loop (docs/CONTINUOUS_LOOP.md §3.1,
§3.2, §7; group D of §9).

    python -m weed_optimizer_framework.tools.collect plan    --config C --out PATH [--classes A,B]
    python -m weed_optimizer_framework.tools.collect fetch   --source ID [--max-bytes B]
    python -m weed_optimizer_framework.tools.collect intake  --source ID
    python -m weed_optimizer_framework.tools.collect names   --source ID --out DIR
    python -m weed_optimizer_framework.tools.collect summary
    python -m weed_optimizer_framework.tools.collect probe

What this module holds, shared by the rest of the package:
  * where the collector's state lives (INC_DIR/intake/, one writer: the
    intake lock) and the format id of every file it writes (FORMATS);
  * the error classes; a Refusal carries a code, an action (hold, refuse or
    close) and the risk class of the person's decision it asks for;
  * canonical JSON, atomic writers, file records, hash-chained append-only
    JSON-lines ledgers, the single-writer lock and the output header.

The collector never trains and never imports a trainer (tests/test_collect_no_train.py).
Nothing in the package names a domain: targets, providers' endpoints, word
lists, lab groups, known items and card tables all come from the domain
config (collect/domains/<domain>.json, format collect-domain/1) and the funnel
domain config it points to by path and sha256.

Standard library only at module level.
"""
from __future__ import annotations

import contextlib
import datetime
import hashlib
import io
import json
import os
import re
import socket
import time
from pathlib import Path

from ..inc import common as C

COLLECT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = COLLECT_DIR.parent
DOMAINS_DIR = COLLECT_DIR / "domains"

FORMATS = {
    "domain": "collect-domain/1",
    "eppo": "collect-eppo/1",
    "candidates": "collect-candidates/1",
    "fetch": "collect-fetch/1",
    "event": "collect-source-event/1",
    "batch": "collect-intake-batch/1",
    "sources": "collect-intake-sources/1",
    "decision": "collect-intake-decision/1",
    "guard": "collect-intake-guard/1",
    "summary": "collect-intake-summary/1",
    "eval_hits": "inc2-eval-hit-sidecar/1",          # intake/<batch>/eval_hits.json (inc2.eval_hits.SIDECAR_FORMAT)
    "state": "collect-state/1",
    "placement": "collect-placement/1",
    "names_cache": "collect-names-cache/1",
    "names": "collect-names/1",
    "pending_names": "collect-pending-names/1",
    "result": "collect-result/1",
}

# Status values of a source in sources.jsonl (contract §3.1 "State").
STATUSES = ("candidate", "held", "fetched", "intaken", "closed", "quarantined")
# What a refusal asks for: hold (wait for a person or a condition), refuse
# (this request is wrong; nothing changes) or close (never retried).
ACTIONS = ("hold", "refuse", "close")


# ------------------------------------------------------------------ errors
class CollectError(RuntimeError):
    """Any refusal of the collector. The CLI maps it to exit code 2."""


class ConfigError(CollectError):
    pass


class ProviderError(CollectError):
    pass


class CredentialsMissing(ProviderError):
    pass


class ByteCapExceeded(ProviderError):
    pass


class ChecksumMismatch(ProviderError):
    pass


class NormaliseError(CollectError):
    pass


class ClassMapError(CollectError):
    pass


class LicenceError(CollectError):
    pass


class StaleInput(CollectError):
    """A recorded input no longer hashes to what its producer recorded."""


class LockHeld(CollectError):
    """Another writer holds INC_DIR/intake/.lock."""


class GuardUnavailable(CollectError):
    """The copy guard (inc2.guard.GuardV2) cannot be loaded: intake fails closed."""


class Refusal(CollectError):
    """A pre-check or gate refusal with a machine-readable reason.

    code     short reason id (e.g. licence_unresolved, image_level_only)
    action   hold | refuse | close (ACTIONS)
    risk     the governance class of the decision a person would take (R3,
             R4) or None when nothing is asked of a person
    failures every pre-check failure found, [{"code", "detail", "action"}]
    """

    def __init__(self, code, detail, action="refuse", risk=None, failures=None):
        if action not in ACTIONS:
            raise ValueError("unknown refusal action %r" % action)
        self.code, self.detail, self.action, self.risk = code, detail, action, risk
        self.failures = list(failures or [{"code": code, "detail": detail, "action": action}])
        super().__init__("%s: %s" % (code, detail))

    def record(self):
        return {"code": self.code, "detail": self.detail, "action": self.action, "risk": self.risk,
                "failures": list(self.failures)}


class NamesPending(Refusal):
    """Class names the offline resolver lacks: lever L26 (collect names) first."""

    def __init__(self, source, names):
        self.names = sorted(set(names))
        super().__init__("names_pending", "%d class name(s) of %s are not in the taxonomy cache or the names "
                         "layer; run `collect names --source %s` (lever L26) first: %s"
                         % (len(self.names), source, source, self.names[:10]), action="hold", risk=None)


# ------------------------------------------------------------------ basics
def utc(now=None):
    t = now if now is not None else datetime.datetime.now(datetime.timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(text):
    return datetime.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc)


def canonical_json(obj):
    """Sorted keys, no spaces, UTF-8; NaN and infinities are refused."""
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError) as e:
        raise CollectError("not canonical JSON (%s)" % e)


def json_text(obj):
    try:
        return json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    except (ValueError, TypeError) as e:
        raise CollectError("not JSON-serialisable (%s)" % e)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj):
    return sha256_text(canonical_json(obj))


def sha256_file(path):
    return C.sha256_file(path)


def atomic_write_bytes(path, data):
    """Write to <name>.tmp beside path, fsync, then os.replace. Returns sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return sha256_bytes(data)


def write_json_atomic(path, obj):
    return atomic_write_bytes(path, json_text(obj).encode("utf-8"))


def write_jsonl_atomic(path, rows):
    buf = io.StringIO()
    for r in rows:
        buf.write(canonical_json(r))
        buf.write("\n")
    return atomic_write_bytes(path, buf.getvalue().encode("utf-8"))


def read_json(path, what=None):
    path = Path(path)
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise CollectError("missing %s %s" % (what or "file", path))
    except (OSError, ValueError) as e:
        raise CollectError("unreadable JSON %s (%s)" % (path, e))


def read_jsonl(path, missing_ok=False):
    path = Path(path)
    if missing_ok and not path.exists():
        return []
    rows = []
    try:
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError as e:
                        raise CollectError("%s line %d is not JSON (%s)" % (path, i, e))
    except FileNotFoundError:
        raise CollectError("missing file %s" % path)
    return rows


def file_record(path):
    path = Path(path)
    if not path.is_file():
        raise CollectError("missing file %s" % path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def check_records(records):
    """Re-hash every {"path", "sha256"} record; StaleInput on the first change."""
    for name in sorted(records or {}):
        rec = records[name]
        if not isinstance(rec, dict) or not rec.get("path"):
            continue
        p = Path(rec["path"])
        if not p.is_file():
            raise StaleInput("input %s (%s) is missing" % (name, p))
        got = sha256_file(p)
        if got != rec.get("sha256"):
            raise StaleInput("input %s (%s) changed: sha256 %s, recorded %s"
                             % (name, p, got[:12], str(rec.get("sha256"))[:12]))


def safe_name(text, keep="._-"):
    """Letters, digits and `keep`; anything else becomes one underscore."""
    s = re.sub(r"[^A-Za-z0-9%s]+" % re.escape(keep), "_", str(text)).strip("_")
    return s or "x"


def seed(text):
    return C.stable_int(text)


def slurm_job_id():
    return os.environ.get("SLURM_JOB_ID") or None


def in_slurm():
    return bool(os.environ.get("SLURM_JOB_ID"))


# ------------------------------------------------------------------ paths
def inc_dir(value=None):
    return Path(value) if value is not None else Path(C.INC_DIR)


def intake_dir(inc=None):
    return inc_dir(inc) / "intake"


def staging_dir(source, inc=None):
    return intake_dir(inc) / "staging" / safe_name(source)


def sources_ledger(inc=None):
    return intake_dir(inc) / "sources.jsonl"


def batches_ledger(inc=None):
    return intake_dir(inc) / "batches.jsonl"


# ------------------------------------------------------------------ code record
def code_record(*modules):
    """{path relative to the tools package: sha256} of every module given."""
    out = {}
    for m in modules:
        f = m if isinstance(m, (str, Path)) else getattr(m, "__file__", None)
        if not f:
            continue
        p = Path(f).resolve()
        try:
            rel = p.relative_to(TOOLS_DIR).as_posix()
        except ValueError:
            rel = p.as_posix()
        out[rel] = sha256_file(p)
    return out


def package_code():
    """sha256 of every module of this package (the job script logs the same)."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(COLLECT_DIR):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for f in sorted(filenames):
            if f.endswith(".py"):
                p = Path(dirpath) / f
                out[p.relative_to(TOOLS_DIR).as_posix()] = sha256_file(p)
    return dict(sorted(out.items()))


def header(kind, cfg, inputs=None, seeds=None, testing=False, code=None):
    """The top-level keys every JSON output of the collector carries: its
    format, the domain, when and where it was built, the configs (path and
    sha256), the code, every input file (path and sha256) and the seeds."""
    if kind not in FORMATS:
        raise CollectError("unknown output kind %r" % (kind,))
    ins = {}
    for name in sorted(inputs or {}):
        v = inputs[name]
        if v is None:
            ins[name] = None
        elif isinstance(v, dict):
            if "sha256" not in v:
                raise CollectError("input record %s lacks sha256" % name)
            ins[name] = dict(v)
        else:
            ins[name] = file_record(v)
    return {
        "format": FORMATS[kind],
        "domain": cfg.name if cfg is not None else None,
        "built_utc": utc(),
        "config": cfg.record() if cfg is not None else None,
        "code": dict(code) if code is not None else package_code(),
        "inputs": ins,
        "seeds": dict(seeds or {}),
        "testing": bool(testing),
        "hostname": socket.gethostname(),
        "slurm_job_id": slurm_job_id(),
    }


VOLATILE_KEYS = ("built_utc", "hostname", "slurm_job_id", "seconds", "ts")


def strip_volatile(obj):
    if isinstance(obj, dict):
        return {k: strip_volatile(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, (list, tuple)):
        return [strip_volatile(v) for v in obj]
    return obj


# ------------------------------------------------------------------ ledgers
EMPTY_SHA256 = sha256_bytes(b"")


_CHAIN_OK = {}          # {path: (size, mtime_ns)} of a ledger this process last verified or appended to


def append_chained(path, row):
    """Append one JSON line to an append-only ledger. The line carries
    prev_sha256, the sha256 of the file's bytes before it, so any edit of an
    earlier line breaks the chain (verify_chain). A ledger whose chain is
    already broken (an edited, reordered or truncated line) is never appended
    to: the state it holds (attempts, closures, holds) cannot be trusted, so
    the writer refuses (StaleInput) until a person repairs it. Returns the row
    written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        st = path.stat()
        key = str(path.resolve())
        if _CHAIN_OK.get(key) != (st.st_size, st.st_mtime_ns):
            probs = verify_chain(path)
            if probs:
                raise StaleInput("the ledger %s has a broken hash chain (%s): nothing is appended; a person "
                                 "repairs it" % (path, "; ".join(probs[:3])))
    prev = sha256_file(path) if path.exists() else EMPTY_SHA256
    row = dict(row, prev_sha256=prev)
    line = (canonical_json(row) + "\n").encode("utf-8")
    with open(path, "ab") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())
    st = path.stat()
    _CHAIN_OK[str(path.resolve())] = (st.st_size, st.st_mtime_ns)
    return row


def verify_chain(path):
    """[] when every line's prev_sha256 is the sha256 of the bytes before it,
    else the problems found (line numbers from 1)."""
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    problems, pos, h = [], 0, hashlib.sha256()
    for i, line in enumerate(data.splitlines(keepends=True), 1):
        want = h.hexdigest()
        try:
            row = json.loads(line.decode("utf-8"))
        except ValueError:
            problems.append("line %d is not JSON" % i)
            break
        if row.get("prev_sha256") != want:
            problems.append("line %d: prev_sha256 %s, the bytes before it hash to %s"
                            % (i, str(row.get("prev_sha256"))[:12], want[:12]))
        h.update(line)
        pos += len(line)
    return problems


# ------------------------------------------------------------------ lock
OWNER_STALE_SECONDS = 9 * 3600      # longer than any collect job runs (run_inc_collect.sh --time=08:00:00)
SQUEUE = "squeue"                   # tests replace it


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError, ValueError, TypeError):
        return True
    return True


def _job_ended(job):
    """True when Slurm no longer lists the job (it ended or was killed at its
    time limit, which runs no release), False when it does, None when squeue
    cannot tell (then the lock is not taken over)."""
    import subprocess
    try:
        p = subprocess.run([SQUEUE, "-h", "-j", str(job), "-o", "%T"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return True if "invalid job id" in (p.stderr or "").lower() else None
    state = (p.stdout or "").strip().upper()
    return not state or state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
                                  "PREEMPTED", "BOOT_FAIL", "DEADLINE")


def _owner_stale(old):
    """Why the owner file of an earlier writer may be taken over, else None."""
    if not isinstance(old, dict):
        return "unreadable owner file"
    if old.get("host") == socket.gethostname() and not _pid_alive(old.get("pid")):
        return "its process %s on this host is gone" % old.get("pid")
    if old.get("job") and _job_ended(old["job"]) is True:
        return "its Slurm job %s has ended" % old["job"]
    try:
        age = time.time() - float(old.get("t") or 0)
    except (TypeError, ValueError):
        age = 0
    if age > OWNER_STALE_SECONDS:
        return "%.1f h old, longer than any collect job runs" % (age / 3600.0)
    return None


@contextlib.contextmanager
def intake_lock(inc=None, what="collect"):
    """The single writer of INC_DIR/intake/ (contract §3.8), refused (LockHeld)
    while another writer lives, never queued. Two locks, as step1_stream's
    writer lock: a non-blocking flock on intake/.lock (one node), and the owner
    file intake/.lock.owner created with O_CREAT|O_EXCL (every node: on
    Lustre, flock is coherent across nodes only on a mount with the 'flock'
    option, and a mount without flock answers ENOLCK or ENOSYS, which is
    tolerated). The owner file of a dead writer (its pid on this host gone,
    its Slurm job ended, or older than any job runs) is taken over."""
    import errno
    d = intake_dir(inc)
    d.mkdir(parents=True, exist_ok=True)
    path = d / ".lock"
    owner = d / ".lock.owner"
    fh = open(path, "a+")
    token = None
    try:
        try:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:  # pragma: no cover - POSIX only in practice
            pass
        except OSError as e:
            if e.errno not in (errno.ENOLCK, errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL,
                               getattr(errno, "ENOTSUP", errno.EOPNOTSUPP)):
                fh.seek(0)
                holder = fh.read().strip()
                raise LockHeld("%s holds %s (%s); one writer at a time" % (holder or "another process", path, what))
        doc = {"token": "%s:%d:%s" % (socket.gethostname(), os.getpid(), os.urandom(8).hex()),
               "host": socket.gethostname(), "pid": os.getpid(), "job": slurm_job_id(), "t": time.time(),
               "utc": utc(), "what": what}
        for _attempt in range(3):
            try:
                fd = os.open(str(owner), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                try:
                    old = json.loads(owner.read_text())
                except (OSError, ValueError):
                    old = None
                why = _owner_stale(old)
                if why is None:
                    raise LockHeld("%s names a live writer (%s); one writer at a time (%s)"
                                   % (owner, {k: (old or {}).get(k) for k in ("what", "host", "pid", "job", "utc")},
                                      what))
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(str(owner))
                continue
            with os.fdopen(fd, "w") as ofh:
                ofh.write(json.dumps(doc, sort_keys=True))
                ofh.flush()
                os.fsync(ofh.fileno())
            token = doc["token"]
            break
        if token is None:
            raise LockHeld("%s could not be taken (%s)" % (owner, what))
        fh.seek(0)
        fh.truncate()
        fh.write("%s pid %d on %s since %s\n" % (what, os.getpid(), socket.gethostname(), utc()))
        fh.flush()
        yield path
    finally:
        if token is not None:
            try:
                if json.loads(owner.read_text()).get("token") == token:
                    os.unlink(str(owner))
            except (OSError, ValueError):
                pass
        try:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        fh.close()
