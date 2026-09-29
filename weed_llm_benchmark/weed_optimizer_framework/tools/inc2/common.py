"""Shared definitions for splits v2 of the continuous loop (docs/CONTINUOUS_LOOP.md
§4.1-4.3, decisions D-A and L-5).

This module re-exports inc.common (the class space, the manifest format,
hashing, materialising, stable_int, dhash) and overrides what v2 changes:

  SPLITS_VERSION "v2", SPLITS_DIR INC_DIR/splits/v2, the v2 LOCK and
  never-train index, BASE_COPIES_INDEX, EVAL_SPLITS (dev, test, imageweeds),
  TRAIN_SPLITS (train_core, tsw22, tsw23), FINAL_EXAMS (dev, imageweeds, test).

Why functions are redefined and not only constants (§4.3 [review]): the v1
functions manifest_path, read_lock, verify_manifest_against_lock and
NeverTrainGuard.load() without an argument read inc.common's own module
globals, so a re-exported call would still resolve v1 paths. Here:

  train_manifest_path(split)   a v2 training manifest (train_core, tsw22, tsw23,
                               base_v2) under SPLITS_DIR
  v2_manifest_path(split)      any manifest LOCK v2 records (the training ones
                               and the byte copies of dev, test, imageweeds)
  eval_manifest_path(split)    an evaluation manifest, explicitly the v1 file:
                               every v2 model is scored by the unchanged v1
                               scorer, which reads the v1 LOCK and exams/v1
  manifest_path(split)         dispatches to one of the two above by split and
                               refuses ood22/ood23 (not v2 splits)
  read_lock_v2(), read_lock_v1()
  read_lock(), verify_manifest_against_lock()
                               refuse: a caller must say which LOCK it means
  NeverTrainGuard.load(path)   a path is required (v1's default is the v1 index);
                               its check() covers the dHash and the 8 flips and
                               rotations (§8), not the dHash alone as in v1
  nevertrain_v2(path=None)     the v2 never-train guard

EXAMS_DIR stays the v1 exam directory for the same reason as the evaluation
manifests.

Decision L-8 (§2.6): a train_core row within 6 bits of an evaluation image
under one of the 8 flips and rotations is dropped from base_v2 and from
train_core.jsonl, through the list TRAIN_CORE_VARIANT_DROPS (sha256 in LOCK
v2). dev, test and imageweeds stay byte copies of v1 (BYTE_COPIES); v2's
train_core is v1's with the listed rows removed (filter_manifest_bytes:
every other line byte-identical, in order), a byte copy when nothing is
listed. Standard library only at module level.
"""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

from ..inc import common as _v1
from ..inc.common import *  # noqa: F401,F403  (the re-export; overrides follow)
from ..inc.common import HOLDOUT_NEAR_DUP_BITS, INC_DIR, dhash, sha256_file

SPLITS_VERSION = "v2"
SPLITS_DIR = INC_DIR / "splits" / SPLITS_VERSION
LOCK_PATH = SPLITS_DIR / "LOCK.json"
NEVER_TRAIN_INDEX = SPLITS_DIR / "nevertrain_dhash.json"
BASE_COPIES_INDEX = SPLITS_DIR / "base_copies_dhash.json"
BASE_MANIFEST = "base_v2"
BASE_PROVENANCE = SPLITS_DIR / "base_v2_provenance.jsonl"
L5_EXCLUDED = SPLITS_DIR / "l5_excluded.jsonl"
# L-8: the train_core rows a flip or rotation puts within 6 bits of an evaluation image
VARIANT_DROPS_NAME = "train_core_variant_drops.jsonl"
TRAIN_CORE_VARIANT_DROPS = SPLITS_DIR / VARIANT_DROPS_NAME
VARIANT_DROPS_LOCK_KEY = "train_core_variant_drops_sha256"
EXAMS_DIR = _v1.EXAMS_DIR                    # the v1 exams: the v1 scorer reads them

EVAL_SPLITS = ("dev", "test", "imageweeds")
TRAIN_SPLITS = ("train_core", "tsw22", "tsw23")
FINAL_EXAMS = ("dev", "imageweeds", "test")
SEALED_EXAM = "test"
V2_TRAIN_MANIFESTS = TRAIN_SPLITS + (BASE_MANIFEST,)
V2_MANIFESTS = TRAIN_SPLITS + (BASE_MANIFEST,) + EVAL_SPLITS
# Manifests v2 holds as byte copies of v1 (§4.1): their sha256 equals v1's LOCK.
BYTE_COPIES = ("dev", "test", "imageweeds")
# Manifests v2 holds as v1's bytes minus the rows L-8 drops (filter_manifest_bytes).
FILTERED_COPIES = ("train_core",)

V1_SPLITS_VERSION = _v1.SPLITS_VERSION
V1_SPLITS_DIR = _v1.SPLITS_DIR
V1_LOCK_PATH = _v1.LOCK_PATH
V1_NEVER_TRAIN_INDEX = _v1.NEVER_TRAIN_INDEX
V1_EVAL_SPLITS = _v1.EVAL_SPLITS
SCORER_MODULE = "weed_optimizer_framework.tools.inc.scorer"
PROTOCOL_PACKAGE = "inc2"


class Inc2Error(RuntimeError):
    """A refusal of the v2 protocol package."""


class AmbiguousV1Call(Inc2Error):
    """A v1 function that reads inc.common's globals was called through inc2."""


# ------------------------------------------------------------------ paths
def train_manifest_path(split):
    """SPLITS_DIR/<split>.jsonl for a v2 training manifest; anything else refuses."""
    if split not in V2_TRAIN_MANIFESTS:
        raise Inc2Error("%r is not a v2 training manifest (%s)" % (split, ", ".join(V2_TRAIN_MANIFESTS)))
    return SPLITS_DIR / ("%s.jsonl" % split)


def v2_manifest_path(split):
    """SPLITS_DIR/<split>.jsonl for any manifest LOCK v2 records."""
    if split not in V2_MANIFESTS:
        raise Inc2Error("%r is not a v2 manifest (%s)" % (split, ", ".join(V2_MANIFESTS)))
    return SPLITS_DIR / ("%s.jsonl" % split)


def eval_manifest_path(split):
    """The v1 manifest of a v2 evaluation split (the v1 scorer's input)."""
    if split not in EVAL_SPLITS:
        raise Inc2Error("%r is not a v2 evaluation split (%s)" % (split, ", ".join(EVAL_SPLITS)))
    return _v1.manifest_path(split)


def manifest_path(split):
    """eval_manifest_path for dev/test/imageweeds, train_manifest_path for the
    training manifests; ood22/ood23 and anything else refuse."""
    if split in EVAL_SPLITS:
        return eval_manifest_path(split)
    if split in V2_TRAIN_MANIFESTS:
        return train_manifest_path(split)
    raise Inc2Error("%r is not a v2 split: evaluation %s, training %s"
                    % (split, list(EVAL_SPLITS), list(V2_TRAIN_MANIFESTS)))


# ------------------------------------------------------------------- locks
def read_lock_v1():
    return _v1.read_lock()


def read_lock_v2(path=None):
    path = Path(path or LOCK_PATH)
    try:
        with open(path) as fh:
            lock = json.load(fh)
    except FileNotFoundError:
        raise Inc2Error("%s missing: splits v2 are not locked" % path)
    if lock.get("splits_version") != SPLITS_VERSION:
        raise Inc2Error("%s is not a v2 LOCK (splits_version %r)" % (path, lock.get("splits_version")))
    return lock


def read_lock(*_a, **_k):
    raise AmbiguousV1Call("inc2.common.read_lock is ambiguous: call read_lock_v2() (training manifests, "
                          "never-train v2) or read_lock_v1() (the scorer's evaluation LOCK)")


def verify_manifest_against_lock(*_a, **_k):
    raise AmbiguousV1Call("inc2.common.verify_manifest_against_lock is ambiguous: call "
                          "verify_manifest_against_lock_v2()")


def verify_manifest_against_lock_v2(split, lock=None):
    """Raise unless SPLITS_DIR/<split>.jsonl hashes to what LOCK v2 recorded."""
    lock = lock or read_lock_v2()
    want = (lock.get("manifests") or {}).get(split)
    if want is None:
        raise Inc2Error("split %r is not in LOCK v2" % split)
    got = sha256_file(v2_manifest_path(split))
    if got != want:
        raise Inc2Error("v2 manifest %s changed since it was locked (%s != %s)" % (split, got[:12], want[:12]))
    return got


# ------------------------------------------------------ L-8 variant drops
def filter_manifest_bytes(data, drop_shas):
    """(bytes, dropped rows) of a JSON-lines manifest's bytes without the rows
    whose image sha256 is in drop_shas. Every other line is kept byte for
    byte, in order, so v2's train_core is v1's minus the L-8 rows and a
    reader can re-derive it from the v1 bytes and the list."""
    drop = {str(s) for s in drop_shas}
    kept, dropped = [], []
    for line in bytes(data).splitlines(keepends=True):
        text = line.strip()
        if text:
            row = json.loads(text)
            if str(row.get("sha256")) in drop:
                dropped.append(row)
                continue
        kept.append(line)
    return b"".join(kept), dropped


def read_variant_drops(lock=None, lock_path=None, production=True):
    """(rows, record) of the L-8 list beside LOCK v2
    (train_core_variant_drops.jsonl), the file hashing to the sha256 LOCK v2
    records. A production LOCK that records no list is refused (every LOCK
    the v2 build writes records one, empty or not); a testing LOCK without
    one gives no rows, recorded. A missing, changed or unreadable file
    refuses (fail closed)."""
    lock_path = Path(lock_path or LOCK_PATH)
    lock = lock if lock is not None else read_lock_v2(lock_path)
    path = lock_path.parent / VARIANT_DROPS_NAME
    want = lock.get(VARIANT_DROPS_LOCK_KEY)
    if want is None:
        if production:
            raise Inc2Error("LOCK v2 %s records no %s: the L-8 drops cannot be checked" % (lock_path,
                                                                                          VARIANT_DROPS_LOCK_KEY))
        return [], {"path": str(path), "sha256": None, "images": 0, "note": "LOCK v2 records none (testing)"}
    try:
        got = sha256_file(path)
    except OSError:
        got = None
    if got != want:
        raise Inc2Error("%s hashes to %s, LOCK v2 records %s" % (path, (got or "none")[:12], str(want)[:12]))
    rows = []
    try:
        with open(path) as fh:
            for ln in fh:
                if ln.strip():
                    r = json.loads(ln)
                    if not isinstance(r, dict) or not r.get("sha256") or not r.get("key"):
                        raise ValueError("a row without key or sha256")
                    rows.append(r)
    except (OSError, ValueError) as e:
        raise Inc2Error("cannot read %s: %s" % (path, e))
    return rows, {"path": str(path), "sha256": got, "images": len(rows), "keys": sorted(r["key"] for r in rows)}


class NeverTrainGuard(_v1.NeverTrainGuard):
    """inc.common.NeverTrainGuard with the v2 never-train definition (§8): an
    image is refused when its dHash OR any of its eight flips and rotations is
    within 6 bits of an evaluation image. The v1 check compares the stored
    dHash only, so a flipped or rotated copy of a test image passed it.

    load() requires the index path: its v1 default is the v1 index, which
    lists every 3SeasonWeedDet10 image."""

    @classmethod
    def load(cls, path=None):
        if path is None:
            raise Inc2Error("NeverTrainGuard.load needs an explicit index path in inc2 "
                            "(nevertrain_v2() loads the v2 index)")
        return super().load(path)

    def check(self, paths, hash_fn=None, variants_fn=None):
        """([(path, split, image, bits)], [unhashable path]) over the dHash and
        the eight variant dHashes of every path (inc2.guard.dhash_variants
        unless variants_fn is given). An image whose variants cannot be
        computed, or whose "id" variant is not its dHash, is unhashable: an
        image the guard cannot compare is one it cannot clear."""
        hash_fn = hash_fn or dhash
        if variants_fn is None:
            from .guard import dhash_variants as variants_fn
        hits, unhashable = [], []
        for p in paths:
            h = hash_fn(p)
            try:
                vs = variants_fn(p)
            except Exception:  # noqa: BLE001 - no variants: unhashable, never a pass
                vs = None
            vals = None
            if h is not None and isinstance(vs, dict) and len(vs) == 8:
                try:
                    vals = [int(v) for v in vs.values()]
                except (TypeError, ValueError):
                    vals = None
                if vals is not None and vs.get("id") is not None and int(vs["id"]) != int(h):
                    vals = None
            if vals is None:
                unhashable.append(str(p))
                continue
            best = None
            for v in [int(h)] + vals:
                m = self.index.find(v)
                if m is not None and (best is None or m[1] < best[1]):
                    best = m
            if best is not None:
                (split, image), bits = best
                hits.append((str(p), split, image, bits))
        return hits, unhashable

    def assert_trainable(self, paths, hash_fn=None, variants_fn=None):
        hits, unhashable = self.check(paths, hash_fn=hash_fn, variants_fn=variants_fn)
        if hits or unhashable:
            raise RuntimeError(
                "never-train guard v2: %d image(s) within %d bits of an evaluation image (dHash or one of its 8 "
                "flips and rotations), %d unhashable; first: %s %s"
                % (len(hits), HOLDOUT_NEAR_DUP_BITS, len(unhashable), hits[:3], unhashable[:3]))
        return True


def nevertrain_v2(path=None):
    """The v2 never-train guard (dev + test + imageweeds). An index that lock
    has not marked complete is refused (min_expected), as in v1."""
    return NeverTrainGuard.load(path or NEVER_TRAIN_INDEX)


# ------------------------------------------------------------ atomic files
def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def json_text(obj):
    """Sorted keys, indent 1, a final newline; NaN is refused."""
    return json.dumps(obj, sort_keys=True, indent=1, allow_nan=False) + "\n"


def _atomic_bytes(path, data):
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
    return sha256_file(path)


def write_bytes_atomic(path, data):
    """Write bytes atomically; returns the file's sha256."""
    return _atomic_bytes(path, data)


def write_csv_atomic(path, header, rows):
    """A CSV (header, then rows as sequences), atomically; returns sha256."""
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(list(header))
    for r in rows:
        w.writerow(["" if v is None else v for v in r])
    return _atomic_bytes(path, buf.getvalue().encode("utf-8"))


def write_json_atomic(path, obj):
    """Write obj as JSON atomically; returns the file's sha256."""
    return _atomic_bytes(path, json_text(obj).encode("utf-8"))


def write_jsonl_atomic(path, rows):
    """One JSON object per line (sorted keys), in the given order; returns sha256."""
    text = "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows)
    return _atomic_bytes(path, text.encode("utf-8"))


def atomic_copy(src, dst):
    """Byte copy src -> dst through a temporary file; returns dst's sha256."""
    with open(src, "rb") as fh:
        return _atomic_bytes(dst, fh.read())


def file_record(path):
    """{"path", "sha256", "bytes"} of an existing file."""
    path = Path(path)
    if not path.is_file():
        raise Inc2Error("missing file %s" % path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
