#!/usr/bin/env python3
"""Tests for the append-only, hash-chained correction channel (WP5).

Covers what the correction channel exists to guarantee:
  * append() refuses a correction with no reason, and refuses a non-human
    author with no verbatim quote, without touching the chain.
  * the single-writer lock: a second writer is refused, not silently dropped.
  * verify() distinguishes five corruption classes with a specific
    first_bad_seq and reason each: a mutated field, a deleted record, a
    reordered record, a truncated tail, a torn final line.
  * a `\\r\\n` / lone `\\r` fixture proves the chain hashes bytes, never a
    decoded string, across a text-mode and a binary-mode read of the same
    file (the documented failure in tools/brain/corpus.py's sha256_bytes).
  * wal.py's crash-safety: an interruption between "recorded intent" and
    "applied" leaves a state `recover()` resolves back to consistent.
  * a static grep: no module other than corrections.py opens a
    corrections.json mirror for writing.

Run:  cd weed_llm_benchmark && python3 -m pytest tests/test_corrections_chain.py -q
"""
import inspect
import json
import os
import pathlib
import re
import stat
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.brain import corrections  # noqa: E402
from weed_optimizer_framework.tools.brain import wal  # noqa: E402

try:
    import fcntl
    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - Windows
    _HAVE_FCNTL = False

TOOLS_ROOT = pathlib.Path(__file__).resolve().parents[1] / "weed_optimizer_framework" / "tools"

DOMAIN = "weed"
HUMAN = "human:harry@x.com"
TIER1 = "tier1:qwen2.5-coder:7b"


def _target(key="round_params.epochs", old=60, new=30):
    return {"key": key, "old": old, "new": new}


def _scope(frm=1, until=None):
    s = {"from_round": frm}
    if until is not None:
        s["until_round"] = until
    return s


def _append_chain(root, n, domain=DOMAIN, author=HUMAN):
    """Append n well-formed records and return the list of records."""
    out = []
    for i in range(1, n + 1):
        r = corrections.append(domain, author, "rev-%d" % i, "set_round_param",
                                _target(new=i), _scope(frm=i), "reason %d" % i,
                                "quote %d" % i, root=root)
        assert r["ok"] is True, r
        out.append(r["record"])
    return out


# --- append() refusals ------------------------------------------------------
def test_append_refuses_without_reason(tmp_path):
    r = corrections.append(DOMAIN, HUMAN, "r1", "k", _target(), _scope(), "", "q", root=tmp_path)
    assert r == {"ok": False, "reason": "reason_required"}
    assert corrections.verify(DOMAIN, root=tmp_path) == {
        "ok": True, "length": 0, "first_bad_seq": None, "reason": None}


def test_append_refuses_tier1_author_without_quote(tmp_path):
    r = corrections.append(DOMAIN, TIER1, "r1", "k", _target(), _scope(), "reason", "",
                            root=tmp_path)
    assert r == {"ok": False, "reason": "quote_required_for_non_human_author"}


def test_append_allows_human_author_without_quote(tmp_path):
    r = corrections.append(DOMAIN, HUMAN, "r1", "k", _target(), _scope(), "reason", "",
                            root=tmp_path)
    assert r["ok"] is True
    assert r["record"]["quote"] == ""


def test_append_refuses_target_without_key(tmp_path):
    r = corrections.append(DOMAIN, HUMAN, "r1", "k", {}, _scope(), "reason", "", root=tmp_path)
    assert r == {"ok": False, "reason": "target_key_required"}


# --- basic chain shape -------------------------------------------------
def test_chain_round_trip_and_linking(tmp_path):
    recs = _append_chain(tmp_path, 3)
    assert [r["seq"] for r in recs] == [1, 2, 3]
    assert recs[0]["prev_hash"] == ""
    assert recs[1]["prev_hash"] == recs[0]["hash"]
    assert recs[2]["prev_hash"] == recs[1]["hash"]
    v = corrections.verify(DOMAIN, root=tmp_path)
    assert v == {"ok": True, "length": 3, "first_bad_seq": None, "reason": None}


def test_verify_of_missing_chain_is_ok_with_zero_length(tmp_path):
    assert corrections.verify("nope", root=tmp_path) == {
        "ok": True, "length": 0, "first_bad_seq": None, "reason": None}


# --- single-writer lock -------------------------------------------------
@pytest.mark.skipif(not _HAVE_FCNTL, reason="advisory lock is POSIX-only")
def test_append_refused_when_lock_held_not_dropped_or_corrupted(tmp_path):
    _append_chain(tmp_path, 1)
    lock_path = corrections._lock_path(DOMAIN, tmp_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    held = open(str(lock_path), "a+")
    fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        r = corrections.append(DOMAIN, HUMAN, "r2", "k", _target(), _scope(), "reason", "",
                                root=tmp_path)
        assert r["ok"] is False
        assert r["reason"] == "locked"
    finally:
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
        held.close()
    # refused, not silently dropped: the chain must be exactly what it was.
    v = corrections.verify(DOMAIN, root=tmp_path)
    assert v["ok"] is True and v["length"] == 1
    # and not corrupted either: a normal append proceeds once the lock frees up.
    r2 = corrections.append(DOMAIN, HUMAN, "r2", "k", _target(), _scope(), "reason", "",
                             root=tmp_path)
    assert r2["ok"] is True and r2["record"]["seq"] == 2


# --- tamper detection: five corruption classes --------------------------
def test_verify_detects_mutated_field(tmp_path):
    _append_chain(tmp_path, 4)
    chain_path = corrections._chain_path(DOMAIN, tmp_path)
    lines = [ln for ln in chain_path.read_bytes().split(b"\n") if ln]
    rec = json.loads(lines[1].decode("utf-8"))          # seq 2
    rec["reason"] = "tampered after the fact, hash left stale"
    lines[1] = json.dumps(rec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    chain_path.write_bytes(b"\n".join(lines) + b"\n")

    v = corrections.verify(DOMAIN, root=tmp_path)
    assert v["ok"] is False
    assert v["reason"] == "mutated_record"
    assert v["first_bad_seq"] == 2
    assert v["length"] == 1


def test_verify_detects_deleted_record(tmp_path):
    _append_chain(tmp_path, 4)
    chain_path = corrections._chain_path(DOMAIN, tmp_path)
    lines = [ln for ln in chain_path.read_bytes().split(b"\n") if ln]
    del lines[1]                                        # remove seq 2 entirely
    chain_path.write_bytes(b"\n".join(lines) + b"\n")

    v = corrections.verify(DOMAIN, root=tmp_path)
    assert v["ok"] is False
    assert v["reason"] == "deleted_record"
    assert v["first_bad_seq"] == 2
    assert v["length"] == 1


def test_verify_detects_reordered_record(tmp_path):
    _append_chain(tmp_path, 4)
    chain_path = corrections._chain_path(DOMAIN, tmp_path)
    lines = [ln for ln in chain_path.read_bytes().split(b"\n") if ln]
    lines[1], lines[2] = lines[2], lines[1]              # swap seq 2 and seq 3
    chain_path.write_bytes(b"\n".join(lines) + b"\n")

    v = corrections.verify(DOMAIN, root=tmp_path)
    assert v["ok"] is False
    assert v["reason"] == "reordered_record"
    assert v["first_bad_seq"] == 3
    assert v["length"] == 1


def test_verify_detects_truncated_tail(tmp_path):
    _append_chain(tmp_path, 5)
    chain_path = corrections._chain_path(DOMAIN, tmp_path)
    lines = [ln for ln in chain_path.read_bytes().split(b"\n") if ln]
    chain_path.write_bytes(b"\n".join(lines[:3]) + b"\n")  # drop seq 4, 5 wholesale

    v = corrections.verify(DOMAIN, root=tmp_path)
    assert v["ok"] is False
    assert v["reason"] == "truncated_tail"
    assert v["first_bad_seq"] == 4
    assert v["length"] == 3


def test_verify_detects_torn_final_line(tmp_path):
    _append_chain(tmp_path, 3)
    chain_path = corrections._chain_path(DOMAIN, tmp_path)
    with open(str(chain_path), "ab") as f:
        f.write(b'{"seq": 4, "prev_hash": "ab12')      # killed mid-write, no newline

    v = corrections.verify(DOMAIN, root=tmp_path)
    assert v["ok"] is False
    assert v["reason"] == "torn_final_line"
    assert v["first_bad_seq"] == 4
    assert v["length"] == 3


def test_torn_final_line_does_not_hide_earlier_tamper(tmp_path):
    """The first bad thing encountered wins, in file order — a torn tail must
    not mask an earlier mutation that verify() would otherwise have caught."""
    _append_chain(tmp_path, 3)
    chain_path = corrections._chain_path(DOMAIN, tmp_path)
    lines = [ln for ln in chain_path.read_bytes().split(b"\n") if ln]
    rec = json.loads(lines[0].decode("utf-8"))
    rec["reason"] = "tampered"
    lines[0] = json.dumps(rec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    chain_path.write_bytes(b"\n".join(lines) + b"\n")
    with open(str(chain_path), "ab") as f:
        f.write(b'{"seq": 4, "garbage')

    v = corrections.verify(DOMAIN, root=tmp_path)
    assert v["reason"] == "mutated_record"
    assert v["first_bad_seq"] == 1


# --- carriage-return byte-hashing fixture -------------------------------
def test_chain_stable_across_text_and_binary_reads(tmp_path):
    """A quote/reason carrying \\r\\n and a lone \\r must not perturb the
    chain: JSON escapes control characters, so the file itself never contains
    a raw \\r acting as a line terminator, and a text-mode read of it must
    re-encode to identical bytes as a binary-mode read. This is the exact
    hazard documented in tools/brain/corpus.py's sha256_bytes (107 of 390
    intact artifacts falsely reported as drifted from hashing a decoded
    string instead of the raw bytes)."""
    quote = "job 44727703.out line 8123:\r\nEpoch 24/60\rTIMEOUT\r\n(walltime)"
    r = corrections.append(DOMAIN, TIER1, "rev-cr", "note", _target(), _scope(),
                            "verbatim job-log excerpt carries CRLF and a lone CR",
                            quote, root=tmp_path)
    assert r["ok"] is True
    assert r["record"]["quote"] == quote

    chain_path = corrections._chain_path(DOMAIN, tmp_path)
    as_binary = chain_path.read_bytes()
    as_text_reencoded = chain_path.read_text(encoding="utf-8").encode("utf-8")
    assert as_binary == as_text_reencoded, (
        "a text-mode read normalised something a binary-mode read did not; "
        "the chain must never contain a raw CR/CRLF acting as a line terminator")

    v = corrections.verify(DOMAIN, root=tmp_path)
    assert v == {"ok": True, "length": 1, "first_bad_seq": None, "reason": None}

    mirror_path = tmp_path / "mirror" / "corrections.json"
    staged = corrections.stage_mirror(DOMAIN, str(mirror_path), root=tmp_path)
    assert staged["ok"] is True
    chk = corrections.check_mirror(DOMAIN, str(mirror_path), root=tmp_path)
    assert chk["ok"] is True and chk["content_ok"] is True


def test_mirror_functions_hash_bytes_never_a_decoded_string():
    """Whitebox guard on this module's own source: check_mirror/stage_mirror
    must never open the mirror in plain text mode, which is exactly the bug
    this module's docstring warns about."""
    for fn in (corrections.check_mirror, corrections.stage_mirror,
               corrections._mirror_payload):
        src = inspect.getsource(fn)
        # a bare text-mode open ("r")/("w") with no "b" flag, as opposed to
        # "rb"/"wb"/"ab" or a call with no literal mode string at all.
        assert not re.search(r'open\([^)]*["\'][rwa]["\']\s*\)', src), (
            "%s appears to open a file in text mode" % fn.__name__)


# --- effective() ---------------------------------------------------------
def test_effective_picks_latest_in_scope_per_key(tmp_path):
    corrections.append(DOMAIN, HUMAN, "r1", "set_round_param",
                        {"key": "epochs", "old": 60, "new": 30},
                        {"from_round": 1, "until_round": 3}, "shrink epochs", "",
                        root=tmp_path)
    corrections.append(DOMAIN, HUMAN, "r2", "set_round_param",
                        {"key": "epochs", "old": 30, "new": 20},
                        {"from_round": 4}, "shrink further", "", root=tmp_path)
    corrections.append(DOMAIN, HUMAN, "r3", "set_round_param",
                        {"key": "batch", "old": 16, "new": 8},
                        {"from_round": 1}, "reduce batch for OOM", "", root=tmp_path)

    eff_r2 = {c["target"]["key"]: c for c in corrections.effective(DOMAIN, 2, root=tmp_path)}
    assert eff_r2["epochs"]["target"]["new"] == 30
    assert eff_r2["batch"]["target"]["new"] == 8

    eff_r4 = {c["target"]["key"]: c for c in corrections.effective(DOMAIN, 4, root=tmp_path)}
    assert eff_r4["epochs"]["target"]["new"] == 20      # r1 expired, r2 took over

    eff_r10 = {c["target"]["key"]: c for c in corrections.effective(DOMAIN, 10, root=tmp_path)}
    assert eff_r10["epochs"]["target"]["new"] == 20     # no until_round -> still active


def test_effective_fails_closed_on_a_broken_chain(tmp_path):
    _append_chain(tmp_path, 3)
    chain_path = corrections._chain_path(DOMAIN, tmp_path)
    lines = [ln for ln in chain_path.read_bytes().split(b"\n") if ln]
    del lines[1]
    chain_path.write_bytes(b"\n".join(lines) + b"\n")
    assert corrections.effective(DOMAIN, 99, root=tmp_path) == []


# --- mirror staging / divergence -----------------------------------------
def test_stage_mirror_is_chmod_0444_and_check_mirror_agrees(tmp_path):
    _append_chain(tmp_path, 3)
    mirror_path = tmp_path / "mirror" / "corrections.json"
    r = corrections.stage_mirror(DOMAIN, str(mirror_path), root=tmp_path)
    assert r["ok"] is True
    mode = stat.S_IMODE(os.stat(str(mirror_path)).st_mode)
    assert mode == 0o444
    chk = corrections.check_mirror(DOMAIN, str(mirror_path), root=tmp_path)
    assert chk == {"ok": True, "content_ok": True, "mode_ok": True,
                   "expected_hash": r["hash"], "actual_hash": r["hash"], "reason": None}


def test_check_mirror_detects_out_of_band_rewrite(tmp_path):
    """A compute-node job rewriting the mirror directly must be visible on
    the very next check_mirror call — both via content and via mode, since a
    rewrite through anything other than stage_mirror will not reproduce the
    0444 stage_mirror always leaves behind."""
    _append_chain(tmp_path, 3)
    mirror_path = tmp_path / "mirror" / "corrections.json"
    corrections.stage_mirror(DOMAIN, str(mirror_path), root=tmp_path)

    os.chmod(str(mirror_path), 0o644)
    mirror_path.write_bytes(b'{"domain": "weed", "length": 999, "tampered": true}')

    chk = corrections.check_mirror(DOMAIN, str(mirror_path), root=tmp_path)
    assert chk["ok"] is False
    assert chk["content_ok"] is False
    assert chk["mode_ok"] is False
    assert chk["reason"] == "mirror_diverged"


def test_check_mirror_missing_file(tmp_path):
    _append_chain(tmp_path, 1)
    chk = corrections.check_mirror(DOMAIN, str(tmp_path / "nope" / "corrections.json"),
                                    root=tmp_path)
    assert chk["ok"] is False
    assert chk["reason"] == "mirror_missing"


def test_stage_mirror_refuses_on_broken_chain(tmp_path):
    _append_chain(tmp_path, 3)
    chain_path = corrections._chain_path(DOMAIN, tmp_path)
    lines = [ln for ln in chain_path.read_bytes().split(b"\n") if ln]
    del lines[1]
    chain_path.write_bytes(b"\n".join(lines) + b"\n")

    mirror_path = tmp_path / "mirror" / "corrections.json"
    r = corrections.stage_mirror(DOMAIN, str(mirror_path), root=tmp_path)
    assert r["ok"] is False
    assert r["reason"] == "chain_broken"
    assert not mirror_path.exists()


# --- wal.py crash safety ---------------------------------------------------
def test_wal_atomic_write_replaces_content(tmp_path):
    target = tmp_path / "state.bin"
    wal.atomic_write(str(target), b"v1")
    wal.atomic_write(str(target), b"v2-longer-content")
    assert target.read_bytes() == b"v2-longer-content"
    assert not list(tmp_path.glob("state.bin.tmp.*"))   # no leftover temp file


def test_wal_recover_reapplies_when_apply_never_happened(tmp_path):
    target = tmp_path / "chain.jsonl"
    wal.atomic_write(str(target), b"line-1\n")
    # simulate a kill between "recorded intent" and "applied": the .wal file
    # is durably written, path is untouched.
    wal.atomic_write(wal._wal_path(str(target)), b"line-1\nline-2\n")
    assert target.read_bytes() == b"line-1\n"

    resolved = wal.recover(str(target))
    assert resolved is True
    assert target.read_bytes() == b"line-1\nline-2\n"
    assert not os.path.exists(wal._wal_path(str(target)))


def test_wal_recover_discards_stale_intent_after_apply_completed(tmp_path):
    target = tmp_path / "chain2.jsonl"
    wal.atomic_write(str(target), b"line-1\n")
    wal.atomic_write(wal._wal_path(str(target)), b"line-1\nline-2\n")
    wal.atomic_write(str(target), b"line-1\nline-2\n")   # apply completed
    # crash lands here, before the final clear step removes the .wal file

    resolved = wal.recover(str(target))
    assert resolved is True
    assert target.read_bytes() == b"line-1\nline-2\n"    # unchanged, not reapplied twice
    assert not os.path.exists(wal._wal_path(str(target)))


def test_wal_recover_is_a_noop_with_nothing_pending(tmp_path):
    target = tmp_path / "chain3.jsonl"
    wal.atomic_write(str(target), b"line-1\n")
    assert wal.recover(str(target)) is False
    assert target.read_bytes() == b"line-1\n"


def test_with_wal_self_heals_a_pending_intent_before_the_next_mutation(tmp_path):
    target = tmp_path / "chain4.jsonl"
    wal.atomic_write(str(target), b"line-1\n")
    wal.atomic_write(wal._wal_path(str(target)), b"line-1\nline-2\n")  # crashed mid with_wal

    result = wal.with_wal(str(target), lambda cur: (cur or b"") + b"line-3\n")
    assert result == b"line-1\nline-2\nline-3\n"
    assert target.read_bytes() == b"line-1\nline-2\nline-3\n"


def test_with_wal_mutate_declining_leaves_path_untouched(tmp_path):
    target = tmp_path / "chain5.jsonl"
    wal.atomic_write(str(target), b"line-1\n")
    result = wal.with_wal(str(target), lambda cur: None)
    assert result is None
    assert target.read_bytes() == b"line-1\n"
    assert not os.path.exists(wal._wal_path(str(target)))


def test_corrections_append_recovers_from_a_wal_crash_leaving_a_consistent_chain(tmp_path):
    """End-to-end: a real corrections chain, a simulated crash landing exactly
    between wal's intent record and its apply, then a plain recover() call
    (as the next call to append() would trigger internally) must leave a
    chain that verify() reports as fully consistent at the intended length."""
    _append_chain(tmp_path, 2)
    chain_path = corrections._chain_path(DOMAIN, tmp_path)
    before = chain_path.read_bytes()

    r3 = corrections.append(DOMAIN, HUMAN, "r3", "k", _target(new=3), _scope(frm=3),
                             "reason 3", "", root=tmp_path)
    assert r3["ok"] is True
    after_three = chain_path.read_bytes()

    # roll the real file back to simulate the apply step never having run,
    # but leave the wal intent as it would have been durably written first.
    chain_path.write_bytes(before)
    wal.atomic_write(wal._wal_path(str(chain_path)), after_three)

    # A pending, recoverable intent is reported as its own reason, distinct
    # from a genuine truncated_tail: the data is not lost, only not yet
    # applied, and wal.recover() (which the next append() calls automatically)
    # is known to fix it.
    v_before = corrections.verify(DOMAIN, root=tmp_path)
    assert v_before == {"ok": False, "length": 2, "first_bad_seq": 3, "reason": "wal_pending"}

    resolved = wal.recover(str(chain_path))
    assert resolved is True

    v_after = corrections.verify(DOMAIN, root=tmp_path)
    assert v_after == {"ok": True, "length": 3, "first_bad_seq": None, "reason": None}
    assert not os.path.exists(wal._wal_path(str(chain_path)))


def test_append_transparently_heals_a_pending_wal_intent(tmp_path):
    """The same scenario, but observed purely through the public append()
    API: a pending intent left by a previous, interrupted append must not
    surface as chain_broken/truncated_tail to the next caller — append()
    resolves it first and then adds its own new record on top."""
    _append_chain(tmp_path, 2)
    chain_path = corrections._chain_path(DOMAIN, tmp_path)
    before = chain_path.read_bytes()

    r3 = corrections.append(DOMAIN, HUMAN, "r3", "k", _target(new=3), _scope(frm=3),
                             "reason 3", "", root=tmp_path)
    assert r3["ok"] is True
    after_three = chain_path.read_bytes()

    chain_path.write_bytes(before)
    wal.atomic_write(wal._wal_path(str(chain_path)), after_three)
    assert corrections.verify(DOMAIN, root=tmp_path)["reason"] == "wal_pending"

    r4 = corrections.append(DOMAIN, HUMAN, "r4", "k", _target(new=4), _scope(frm=4),
                             "reason 4", "", root=tmp_path)
    assert r4["ok"] is True
    assert r4["record"]["seq"] == 4          # seq 3 was healed in first, then 4 appended
    v = corrections.verify(DOMAIN, root=tmp_path)
    assert v == {"ok": True, "length": 4, "first_bad_seq": None, "reason": None}


# --- static single-writer invariant --------------------------------------
_WRITE_HINTS = re.compile(
    r'open\([^)]*["\'][wax]b?["\']|\bos\.chmod\(|\bwal\.atomic_write\(|\.write_bytes\(|'
    r'\bwrite_json\(')


def _flags_a_write_near(text, needle="corrections.json", window=3):
    lines = text.splitlines()
    hits = []
    for i, line in enumerate(lines):
        if needle not in line:
            continue
        lo, hi = max(0, i - window), min(len(lines), i + window + 1)
        if _WRITE_HINTS.search("\n".join(lines[lo:hi])):
            hits.append(i + 1)
    return hits


def test_static_detector_flags_a_synthetic_violation(tmp_path):
    """Proves the detector below actually works, independent of whatever is
    currently in the tree: a file that plainly opens the mirror for writing
    must be flagged."""
    bad = tmp_path / "rogue_worker.py"
    bad.write_text(
        'def rewrite_mirror(base):\n'
        '    p = base + "/corrections.json"\n'
        '    with open(p, "w") as f:\n'
        '        f.write("{}")\n')
    assert _flags_a_write_near(bad.read_text()) != []


def test_static_no_other_module_writes_the_mirror():
    """Regression guard: only corrections.py may open the corrections.json
    mirror for writing. This must fail the moment another worker module adds
    such a write, because that write would bypass every guarantee
    check_mirror/stage_mirror give the rest of the system."""
    offenders = []
    for path in TOOLS_ROOT.rglob("*.py"):
        if path.resolve() == pathlib.Path(corrections.__file__).resolve():
            continue
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "corrections.json" not in text:
            continue
        for lineno in _flags_a_write_near(text):
            offenders.append("%s:%d" % (path, lineno))
    assert offenders == [], "modules writing the corrections mirror: %s" % offenders


def test_hashing_agrees_with_the_corpus_exporter():
    """The two hashers must not drift, and both must hash BYTES.

    corrections.py reimplemented these helpers rather than importing them,
    because corpus.py was being edited concurrently. That is a reasonable
    reason to copy and no reason at all to leave the copy unchecked: the
    exporter's own integrity check once reported 107 of 390 intact artifacts as
    drifted because text-mode reading normalised the carriage returns that job
    logs are full of. If either copy regresses to hashing a decoded string, this
    fails.
    """
    from weed_optimizer_framework.tools.brain import corpus
    from weed_optimizer_framework.tools.brain import corrections as C

    fixtures = [
        b"plain line\n",
        b"windows line\r\n",
        b"bare carriage return\rand more",
        b"\xef\xbb\xbfbyte order mark",
        b"mixed\r\nand\rand\n",
        b"",
    ]
    for raw in fixtures:
        assert C._sha256_bytes(raw) == corpus.sha256_bytes(raw), repr(raw)

    # A decoded string and its bytes must NOT collide once a CR is involved,
    # which is what proves the hash is taken over bytes.
    crlf, lf = b"a\r\nb", b"a\nb"
    assert C._sha256_bytes(crlf) != C._sha256_bytes(lf)

    for obj in ({"b": 1, "a": 2}, {"a": 2, "b": 1}, {"nested": {"y": [1, 2], "x": None}}):
        assert C._canonical(obj) == corpus.canonical_json(obj), obj
