"""Append-only, hash-chained correction channel (WP5, v3.25.x).

What a "correction" is
-----------------------
A recorded override of one round parameter or one gate decision, with a
reason and (for anything not a person) a verbatim quote of the evidence that
justified it: `{seq, prev_hash, hash, domain, author, review_id, kind,
target{key,old,new}, scope{from_round,until_round}, reason, quote, ts}`. The
scheduler and the dashboard read `effective()` to know what is currently in
force for a round; nothing here decides whether a correction is a GOOD one —
`tools/brain/policy.py` and the review flow decide that before `append()` is
ever called. This module's only job is to make the record of what was decided
tamper-evident and to refuse the two ways that record can lie.

Refusal #1 — no reason. A correction with no stated reason is indistinguishable
from a silent config edit; `append()` refuses it before it reaches the chain.

Refusal #2 — an LLM-authored correction with no verbatim quote. `human:*`
authors are exempt (a person is accountable for their own words); every other
author string (`tier0:*`, `tier1:*`, `tier2:*`, `round-scheduler`) must carry
a quote of the artifact line that justified the correction. An unquoted
model-authored override is exactly the failure this channel exists to catch:
a plausible-sounding correction with nothing behind it that a reviewer, months
later, can check against a real artifact.

Hash chain
----------
Same shape as `tools/brain/trace.py`'s chain (`hash` here plays the role of
`sha` there, `prev_hash` the role of `sha_prev`), because a reader that already
knows how to read one chain should not have to learn a second convention. The
difference is deliberate: `trace.py` is written by many uncoordinated SLURM
jobs appending to their own per-step file with O_APPEND, so its worst case is
a torn final line and its chain tolerates that by design. This channel has
exactly one writer for a whole domain (see SINGLE WRITER below) and every
write goes through `wal.with_wal`, so a self-inflicted torn line should never
happen — `verify()` still detects one, because "should never happen" is not
the same claim as "cannot happen", and because a chain-integrity checker that
only handles the failure modes its own writer produces is not actually
checking for tampering.

Every hash in this module is taken over BYTES, never over a string a text-mode
read has touched. `tools/brain/corpus.py`'s `sha256_bytes` documents why: a
job log carries carriage returns from progress bars, and text-mode reading
normalises `\r\n` and lone `\r` to `\n`, which made 107 of 390 intact
artifacts in that module report as drifted before the fix. This module never
imports that one (importing a module several other WP5 pieces are actively
editing would couple this file's tests to their in-flight state), but it
follows the same rule for the same documented reason: chain files are read
with `"rb"`, split on `b"\n"` (never text `.splitlines()`, which treats a bare
`\r` as its own line break too), and the mirror in `check_mirror` is compared
byte-for-byte against a freshly rebuilt copy, never against a decoded string.

SINGLE WRITER
-------------
Only the lab round-scheduler thread may call `append()` for a given domain.
Enforcement is an advisory `fcntl.flock(LOCK_EX | LOCK_NB)` on a
`corrections.lock` sidecar next to the chain: a second writer does not block
and does not silently lose its write — `append()` returns `{"ok": False,
"reason": "locked"}` and touches nothing. A caller that needs to know why a
correction did not land checks `ok`; nothing here raises.

Crash safety
------------
Every write to the chain file goes through `tools/brain/wal.py`'s
`with_wal`, for the same reason `wal.py` itself exists: a SLURM walltime kill
(2026-08-29, jobs 44727703 / 44767709) can land between "decided the next
record" and "wrote it", and a correction silently lost that way is worse than
one that was never proposed, because the round it was meant to fix keeps
running on the un-corrected parameter. `append()` resolves any pending intent
(`wal.recover`) before it ever reads the chain's current state, so a domain
whose previous writer was killed mid-append self-heals on the next call
without needing an operator to notice first.

The mirror
----------
`stage_mirror` writes a read-only (`chmod 0444`) snapshot of the chain to a
path the dashboard and cluster-side jobs read without replaying the whole
chain. `check_mirror` re-derives what that snapshot should currently contain
and compares it byte-for-byte (plus the file's mode) against what is actually
on disk, so a compute-node job that rewrites the mirror directly — bypassing
this module, and therefore bypassing every guarantee above — is detectable
from `check_mirror`'s return value alone, on the very next call.
"""
import hashlib
import json
import sys
import os
import pathlib
import re
import stat
import time

try:                                     # package import (the normal path)
    from . import wal
except ImportError:                      # direct execution from this directory
    import wal                           # type: ignore

try:
    import fcntl                         # POSIX only (lab + cluster are Linux, dev is macOS)
    _HAVE_FCNTL = True
except ImportError:                      # pragma: no cover - Windows fallback
    _HAVE_FCNTL = False

TOOL_VERSION = "wp5-corrections/1"

# Same sanitiser tools/brain/evidence.py uses for a domain name that ends up
# in a filesystem path: keep it, never invent a second convention for the
# same problem.
_DOM_RE = re.compile(r"[^a-z0-9_]")

# Authors exempt from the verbatim-quote requirement. Everything else
# (round-scheduler, tier0:*, tier1:*, tier2:*) must carry a quote.
_QUOTE_EXEMPT_PREFIX = "human:"

_CHAIN_NAME = "corrections.jsonl"
_CHECKPOINT_NAME = "corrections.checkpoint.json"
_LOCK_NAME = "corrections.lock"


# --- paths -------------------------------------------------------------
def _domain(domain):
    return _DOM_RE.sub("", str(domain or "").strip().lower())[:40] or "unknown"


def _repo_root():
    return os.environ.get("REPO_ROOT") or os.path.expanduser("~/weed_llm_benchmark")


def _domain_dir(dom, root=None):
    base = pathlib.Path(str(root)) if root else pathlib.Path(_repo_root())
    return base / "results" / "framework" / "_brain" / dom


def _chain_path(dom, root=None):
    return _domain_dir(dom, root) / _CHAIN_NAME


def _checkpoint_path(dom, root=None):
    return _domain_dir(dom, root) / _CHECKPOINT_NAME


def _lock_path(dom, root=None):
    return _domain_dir(dom, root) / _LOCK_NAME


# --- canonical form / hashing -------------------------------------------
def _canonical(obj):
    """Compact, key-sorted, ASCII JSON — the only form anything here hashes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, default=str)


def _sha256_bytes(b):
    """sha256 over raw bytes. Never pass this a str that came from a text-mode
    file read — encode explicitly instead, so the byte sequence hashed is
    exactly the byte sequence on disk."""
    if not isinstance(b, bytes):
        b = str(b).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _record_hash_from_body(rec):
    """The hash a record with this content (ignoring any stored `hash`) should
    carry. Round-tripping through `_canonical` before hashing (done once at
    write time, in `append`, and implicitly here at read time since `rec`
    already came from `json.loads`) means the digest always covers exactly
    what a reader parses back, never a Python object the writer held that
    JSON cannot represent exactly (e.g. a tuple, or a float repr difference)."""
    stripped = {k: v for k, v in rec.items() if k != "hash"}
    return _sha256_bytes(_canonical(stripped).encode("utf-8"))


# --- chain I/O -----------------------------------------------------------
def _read_chain_bytes(dom, root):
    try:
        with open(str(_chain_path(dom, root)), "rb") as f:
            return f.read()
    except OSError:
        return b""


def _read_checkpoint(dom, root):
    try:
        with open(str(_checkpoint_path(dom, root)), "rb") as f:
            obj = json.loads(f.read().decode("utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _write_checkpoint(dom, root, seq, h):
    """Record the high-water mark (seq, hash) reached by the last successful
    append. The chain's own hash links cannot, by themselves, tell a whole
    trailing record being deleted apart from a writer that simply had not got
    there yet — both leave an internally consistent prefix. This checkpoint is
    the external memory that makes the difference visible: a chain that ends
    short of what this file remembers has lost records `verify()` cannot see
    any other way. A crash between the chain write and this one only ever
    leaves the checkpoint stale (lower than reality), never wrongly ahead, so
    the checkpoint alone can under-detect a truncation in that narrow window
    but never manufacture a false one.
    """
    body = _canonical({"seq": int(seq), "hash": str(h)}).encode("utf-8")
    wal.atomic_write(str(_checkpoint_path(dom, root)), body)


# --- the chain walk (shared by verify / append / effective / mirror) -----
def _walk(raw, checkpoint):
    """Parse and validate chain bytes. Returns {ok, length, first_bad_seq,
    reason, last_hash, records}. `records` holds every record verified good,
    in order, up to the first problem (or all of them, if none).

    Detection is layered so each corruption class gets its own reason:

      * a record is parsed and self-checked (its own `hash` against its own
        content) independently of its neighbours, so a MUTATED field is
        caught at the record that was actually changed, never blamed on the
        one after it.
      * a chain-link break (`prev_hash` does not match the previous good
        record's `hash`) is then classified by whether the hash it claims as
        its predecessor exists ANYWHERE among the file's self-valid records:
        if it does, that record's true predecessor was moved elsewhere in the
        file (REORDERED); if it does not exist at all, a record that used to
        sit there is gone (DELETED).
      * the final line is checked for a missing trailing newline before it is
        even parsed — the same tell `trace.py` uses for a writer killed
        mid-write (TORN final line) — because a half-written JSON object can
        coincidentally still fail to parse for reasons unrelated to being
        torn, and the newline check is the one signal that is unambiguous.
      * a wholly-missing trailing suffix of otherwise-untouched, self-valid
        records (TRUNCATED tail) leaves no internal trace at all — the prefix
        that remains is perfectly consistent — so it is only visible against
        the external checkpoint recorded by the last successful `append`.
    """
    result = {"ok": True, "length": 0, "first_bad_seq": None, "reason": None,
              "last_hash": "", "records": []}
    if not raw:
        return result
    ends_nl = raw.endswith(b"\n")
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        lines = lines[:-1]
    if not lines:
        return result

    parsed = []             # list of (status, record_or_None)
    all_hashes = set()
    for idx, raw_line in enumerate(lines):
        is_last = idx == len(lines) - 1
        if is_last and not ends_nl:
            parsed.append(("torn", None))
            continue
        try:
            rec = json.loads(raw_line.decode("utf-8"))
        except Exception:
            parsed.append(("unparseable", None))
            continue
        if not isinstance(rec, dict) or "hash" not in rec:
            parsed.append(("unparseable", None))
            continue
        if not isinstance(rec.get("hash"), str) or rec["hash"] != _record_hash_from_body(rec):
            parsed.append(("hash_mismatch", rec))
            continue
        parsed.append(("ok", rec))
        all_hashes.add(rec["hash"])

    reasons = {"torn": "torn_final_line", "unparseable": "unparseable_record",
               "hash_mismatch": "mutated_record"}
    expected_prev = ""
    expected_seq = 1
    good = []
    for status, rec in parsed:
        if status != "ok":
            bad_seq = expected_seq
            if isinstance(rec, dict) and isinstance(rec.get("seq"), int):
                bad_seq = rec["seq"]
            result.update(ok=False, length=len(good), first_bad_seq=bad_seq,
                          reason=reasons[status], last_hash=expected_prev, records=good)
            return result
        prev_hash = rec.get("prev_hash")
        claimed_seq = rec.get("seq")
        if prev_hash != expected_prev:
            if prev_hash in all_hashes:
                bad_seq = claimed_seq if isinstance(claimed_seq, int) else expected_seq
                result.update(ok=False, length=len(good), first_bad_seq=bad_seq,
                              reason="reordered_record", last_hash=expected_prev, records=good)
            else:
                result.update(ok=False, length=len(good), first_bad_seq=expected_seq,
                              reason="deleted_record", last_hash=expected_prev, records=good)
            return result
        if claimed_seq != expected_seq:
            result.update(ok=False, length=len(good), first_bad_seq=expected_seq,
                          reason="sequence_mismatch", last_hash=expected_prev, records=good)
            return result
        good.append(rec)
        expected_prev = rec["hash"]
        expected_seq += 1

    length = len(good)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("seq"), int) \
            and checkpoint["seq"] > length:
        result.update(ok=False, length=length, first_bad_seq=length + 1,
                      reason="truncated_tail", last_hash=expected_prev, records=good)
        return result
    result.update(ok=True, length=length, first_bad_seq=None, reason=None,
                  last_hash=expected_prev, records=good)
    return result


def verify(domain, root=None):
    """Recompute the whole chain for `domain`. Returns {ok, length,
    first_bad_seq, reason}. A missing or empty chain is `ok` with length 0 —
    a domain with no corrections yet is a different fact from a chain that
    was tampered with, and callers tell them apart by `length`.

    A pending `wal.py` intent (a writer killed between "decided" and
    "written") is reported as its own `wal_pending` reason rather than as
    `truncated_tail`: the checkpoint left by the interrupted write correctly
    notices the chain file is short of what was decided, but that data is not
    lost — a `wal.recover()` call (which `append()` makes automatically
    before every write) puts it back. Collapsing that into `truncated_tail`
    would tell an operator to go hunting for a corruption that a recover call
    already knows how to fix.
    """
    dom = _domain(domain)
    chain_path = _chain_path(dom, root)
    raw = _read_chain_bytes(dom, root)
    ck = _read_checkpoint(dom, root)
    w = _walk(raw, ck)

    intent_path = wal._wal_path(str(chain_path))
    try:
        with open(intent_path, "rb") as f:
            intent = f.read()
    except OSError:
        intent = None
    if intent is not None and intent != raw:
        return {"ok": False, "length": w["length"], "first_bad_seq": w["length"] + 1,
                "reason": "wal_pending"}

    return {"ok": w["ok"], "length": w["length"], "first_bad_seq": w["first_bad_seq"],
            "reason": w["reason"]}


# --- append ----------------------------------------------------------------
def append(domain, author, review_id, kind, target, scope, reason, quote, root=None):
    """Append one correction. Returns {"ok": True, "record": {...}} or a
    structured refusal {"ok": False, "reason": ..., ...} — this never raises
    and never partially writes; a refused correction leaves the chain exactly
    as it was.

    Refusals, in the order checked: no `reason`; a non-`human:*` `author`
    with no `quote`; `target` with no `key`; the single-writer lock already
    held by another process; the existing chain not verifying (a new
    correction is not trustworthy appended on top of one that already is
    not). Only past all of those does a record get built, hashed and written
    through `wal.with_wal`.
    """
    if not reason or not str(reason).strip():
        return {"ok": False, "reason": "reason_required"}
    author = str(author or "").strip()
    if not author:
        return {"ok": False, "reason": "author_required"}
    if not author.startswith(_QUOTE_EXEMPT_PREFIX) and not (quote and str(quote).strip()):
        return {"ok": False, "reason": "quote_required_for_non_human_author"}
    if not isinstance(target, dict) or not target.get("key"):
        return {"ok": False, "reason": "target_key_required"}

    dom = _domain(domain)
    chain_path = _chain_path(dom, root)
    lock_path = _lock_path(dom, root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_f = open(str(lock_path), "a+")
    try:
        if _HAVE_FCNTL:
            try:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return {"ok": False, "reason": "locked",
                        "detail": "corrections lock for domain %r is held by another writer" % dom}
        # Resolve any intent left by a writer killed between "decided" and
        # "written" before this call reads the chain's current state, so a
        # domain whose previous append was interrupted self-heals here rather
        # than reporting a false chain_broken / truncated_tail on top of a
        # perfectly recoverable state.
        wal.recover(str(chain_path))

        raw = _read_chain_bytes(dom, root)
        ck = _read_checkpoint(dom, root)
        w = _walk(raw, ck)
        if not w["ok"]:
            return {"ok": False, "reason": "chain_broken",
                    "verify": {"ok": w["ok"], "length": w["length"],
                               "first_bad_seq": w["first_bad_seq"], "reason": w["reason"]}}

        raw_record = {
            "seq": w["length"] + 1,
            "prev_hash": w["last_hash"],
            "domain": dom,
            "author": author,
            "review_id": str(review_id or ""),
            "kind": str(kind or ""),
            "target": dict(target),
            "scope": dict(scope or {}),
            "reason": str(reason),
            "quote": str(quote or ""),
            "ts": time.time(),
        }
        # One JSON round trip before hashing, same reason trace.py does it:
        # the digest must cover exactly what a reader parses back, not a
        # Python object a reader never sees in that shape.
        body = json.loads(_canonical(raw_record))
        body["hash"] = _record_hash_from_body(body)
        line = (_canonical(body) + "\n").encode("utf-8")

        def _mutate(current):
            return (current or b"") + line

        new_bytes = wal.with_wal(str(chain_path), _mutate)
        if new_bytes is None:
            return {"ok": False, "reason": "write_failed"}
        _write_checkpoint(dom, root, body["seq"], body["hash"])
        return {"ok": True, "record": body}
    finally:
        if _HAVE_FCNTL:
            try:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        lock_f.close()


# --- effective corrections for a round -------------------------------------
def effective(domain, round_no, root=None):
    """Corrections currently in force for `round_no`: for each `target.key`,
    the highest-`seq` correction whose `scope.from_round <= round_no` and
    (no `until_round`, or `round_no <= until_round`). Returns [] both when
    there is nothing in scope and when the chain does not verify — a
    tampered chain must never be honoured, so this fails closed rather than
    returning whatever prefix happened to still look consistent.
    """
    dom = _domain(domain)
    raw = _read_chain_bytes(dom, root)
    ck = _read_checkpoint(dom, root)
    w = _walk(raw, ck)
    if not w["ok"]:
        return []
    try:
        rn = int(round_no)
    except (TypeError, ValueError):
        return []

    by_key = {}
    for rec in w["records"]:
        scope = rec.get("scope") or {}
        try:
            frm = int(scope.get("from_round", 0))
        except (TypeError, ValueError):
            frm = 0
        if frm > rn:
            continue
        until = scope.get("until_round")
        if until is not None:
            try:
                if rn > int(until):
                    continue
            except (TypeError, ValueError):
                pass
        key = (rec.get("target") or {}).get("key")
        if key is None:
            continue
        prev = by_key.get(key)
        if prev is None or rec.get("seq", 0) > prev.get("seq", 0):
            by_key[key] = rec
    return sorted(by_key.values(), key=lambda r: r.get("seq", 0))


# --- read-only mirror --------------------------------------------------
def _mirror_payload(dom, root):
    raw = _read_chain_bytes(dom, root)
    ck = _read_checkpoint(dom, root)
    w = _walk(raw, ck)
    payload = {"domain": dom, "length": w["length"], "last_hash": w["last_hash"],
               "records": w["records"]}
    return w, payload


def stage_mirror(domain, path, root=None):
    """Write a read-only snapshot of `domain`'s chain to `path` (chmod 0444).

    Refuses (writes nothing) when the live chain does not currently verify —
    a mirror is a read-optimised copy other processes trust without replaying
    the chain themselves, so it must never publish a state this module would
    not itself stand behind. Returns {"ok": False, "reason": "chain_broken",
    "verify": {...}} in that case, {"ok": True, "path", "length", "hash"}
    otherwise.
    """
    dom = _domain(domain)
    w, payload = _mirror_payload(dom, root)
    if not w["ok"]:
        return {"ok": False, "reason": "chain_broken",
                "verify": {"ok": w["ok"], "length": w["length"],
                           "first_bad_seq": w["first_bad_seq"], "reason": w["reason"]}}
    body = _canonical(payload).encode("utf-8")
    p = pathlib.Path(str(path))
    if p.exists():
        try:
            os.chmod(str(p), 0o644)   # drop read-only so the replace below can land
        except OSError:
            pass
    wal.atomic_write(str(p), body)
    os.chmod(str(p), 0o444)
    return {"ok": True, "path": str(p), "length": w["length"], "hash": _sha256_bytes(body)}


def check_mirror(domain, path, root=None):
    """Divergence report for the mirror at `path` against the live chain,
    right now. Returns {"ok", "content_ok", "mode_ok", "expected_hash",
    "actual_hash", "reason"}.

    `content_ok` compares BYTES read from disk (`"rb"`, never a decoded
    string — see the module docstring) against the bytes this domain's chain
    would produce if staged again this instant. `mode_ok` is the cheaper,
    faster tell: `stage_mirror` always leaves the file at 0444, so anything
    that replaced the file out of band (a compute-node job overwriting the
    shared path directly, say) is visible in the mode alone, before content
    is even compared. `ok` is False if the chain itself does not verify,
    regardless of the mirror — a mirror that faithfully reflects a tampered
    chain is not a state to trust either.
    """
    dom = _domain(domain)
    w, payload = _mirror_payload(dom, root)
    expected = _canonical(payload).encode("utf-8")
    p = pathlib.Path(str(path))
    if not p.exists():
        return {"ok": False, "reason": "mirror_missing", "content_ok": False,
                "mode_ok": False, "expected_hash": _sha256_bytes(expected), "actual_hash": None}
    with open(str(p), "rb") as f:
        actual = f.read()
    content_ok = actual == expected
    try:
        mode_ok = stat.S_IMODE(os.stat(str(p)).st_mode) == 0o444
    except OSError:
        mode_ok = False
    if not w["ok"]:
        reason = "chain_broken"
    elif not content_ok:
        reason = "mirror_diverged"
    elif not mode_ok:
        reason = "mirror_not_readonly"
    else:
        reason = None
    return {"ok": w["ok"] and content_ok and mode_ok, "content_ok": content_ok,
            "mode_ok": mode_ok, "expected_hash": _sha256_bytes(expected),
            "actual_hash": _sha256_bytes(actual), "reason": reason}


# --- CLI (read-only by design) ----------------------------------------------

def _main(argv):
    """Inspect the chain from a shell. There is deliberately no `append` here.

    Appending is single-writer: only the lab scheduler thread holds the lock, and
    a shell command that took it would be a second writer by definition. The
    commands below read. During an incident the questions are "is the chain
    intact", "what is in force for this round" and "has the mirror been touched",
    and an append-only ledger nobody can inspect from a terminal is one nobody
    audits.
    """
    import argparse
    ap = argparse.ArgumentParser(prog="corrections")
    ap.add_argument("--root", default=None)
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="recompute the whole chain")
    v.add_argument("domain")
    e = sub.add_parser("effective", help="corrections in force for a round")
    e.add_argument("domain")
    e.add_argument("round_no", type=int)
    m = sub.add_parser("check-mirror", help="compare the staged mirror to the chain")
    m.add_argument("domain")
    m.add_argument("path")
    try:
        a = ap.parse_args(argv[1:])
    except SystemExit as exc:
        return int(exc.code or 0)

    if a.cmd == "verify":
        rep = verify(a.domain, root=a.root)
        print(json.dumps(rep, indent=1, sort_keys=True, default=str))
        return 0 if rep.get("ok") else 1
    if a.cmd == "effective":
        rows = effective(a.domain, a.round_no, root=a.root)
        print(json.dumps(rows, indent=1, sort_keys=True, default=str))
        return 0
    if a.cmd == "check-mirror":
        rep = check_mirror(a.domain, a.path, root=a.root)
        print(json.dumps(rep, indent=1, sort_keys=True, default=str))
        return 0 if rep.get("ok") else 1
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
